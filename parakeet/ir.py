"""The low-level typed IR that every backend consumes.

By the time a function reaches this form all Python-level structure is gone:
closures, tuples, slices, array views and higher-order operators have been
resolved by the frontend.  What remains is a C-like program over typed scalars
and flat buffers, with one parallel construct (:class:`ParLoop`).

Expressions are immutable trees; statements are mutable so the frontend can
patch them (e.g. hoist a materialisation) while it elaborates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from .errors import SourceLocation
from .ndtypes import Bool, BufType, Int64, ScalarType

# --------------------------------------------------------------------------
# expressions
# --------------------------------------------------------------------------


class Expr:
  __slots__ = ("type",)
  type: ScalarType | BufType


class Const(Expr):
  __slots__ = ("value",)

  def __init__(self, value, type: ScalarType):
    if type.kind == "b":
      value = bool(value)
    elif type.kind == "f":
      value = float(value)
    else:
      value = int(value)
    self.value = value
    self.type = type

  def __repr__(self) -> str:
    return f"{self.value!r}:{self.type}"


class Var(Expr):
  __slots__ = ("name", "uid")
  _counter = 0

  def __init__(self, name: str, type: ScalarType | BufType):
    Var._counter += 1
    self.uid = Var._counter
    self.name = f"{name}_{self.uid}"
    self.type = type

  def __repr__(self) -> str:
    return self.name


class Prim(Expr):
  """A primitive scalar operation; all operands already have their final types."""

  __slots__ = ("op", "args")

  def __init__(self, op: str, args: tuple[Expr, ...], type: ScalarType):
    self.op = op
    self.args = args
    self.type = type

  def __repr__(self) -> str:
    return f"{self.op}({', '.join(map(repr, self.args))})"


class Cast(Expr):
  __slots__ = ("arg",)

  def __init__(self, arg: Expr, type: ScalarType):
    self.arg = arg
    self.type = type

  def __repr__(self) -> str:
    return f"{self.type}({self.arg!r})"


class Load(Expr):
  __slots__ = ("buf", "index")

  def __init__(self, buf: Var, index: Expr):
    self.buf = buf
    self.index = index
    self.type = buf.type.elt  # type: ignore[union-attr]

  def __repr__(self) -> str:
    return f"{self.buf!r}[{self.index!r}]"


class Null(Expr):
  """The empty buffer reference."""

  __slots__ = ()

  def __init__(self, type: BufType):
    self.type = type

  def __repr__(self) -> str:
    return "NULL"


class RtCall(Expr):
  """A call into the runtime (thread count, RNG key, ...); host-side only
  unless ``pure`` (pure calls are plain functions available on every device)."""

  __slots__ = ("name", "args", "pure")

  def __init__(self, name: str, args: tuple[Expr, ...], type: ScalarType, pure: bool = False):
    self.name = name
    self.args = args
    self.type = type
    self.pure = pure

  def __repr__(self) -> str:
    return f"@{self.name}({', '.join(map(repr, self.args))})"


def children(e: Expr) -> tuple[Expr, ...]:
  if isinstance(e, (Prim, RtCall)):
    return e.args
  if isinstance(e, Cast):
    return (e.arg,)
  if isinstance(e, Load):
    return (e.buf, e.index)
  return ()


def walk_expr(e: Expr) -> Iterator[Expr]:
  stack = [e]
  while stack:
    cur = stack.pop()
    yield cur
    stack.extend(children(cur))


def is_trivial(e: Expr) -> bool:
  return isinstance(e, (Const, Var))


def same(a: Expr, b: Expr) -> bool:
  """Structural equality (used to recognise identical views)."""
  if a is b:
    return True
  if type(a) is not type(b) or a.type is not b.type:
    return False
  if isinstance(a, Const):
    return a.value == b.value  # type: ignore[attr-defined]
  if isinstance(a, Var):
    return False
  if isinstance(a, (Prim, RtCall)):
    if (a.op if isinstance(a, Prim) else a.name) != (
      b.op if isinstance(b, Prim) else b.name  # type: ignore[attr-defined]
    ):
      return False
  ca, cb = children(a), children(b)
  return len(ca) == len(cb) and all(same(x, y) for x, y in zip(ca, cb))


# --------------------------------------------------------------------------
# statements
# --------------------------------------------------------------------------


@dataclass(eq=False)
class ErrorInfo:
  """A runtime failure the generated code can report."""

  exc: str  # name of a built-in exception class
  fmt: str  # message; "{0}", "{1}", ... are filled with int64 runtime values
  loc: SourceLocation | None = None


@dataclass(eq=False)
class Scope:
  """A region (function body or one loop iteration) that owns allocations.

  Buffers allocated inside a scope are released when the scope exits, unless
  they were handed to a longer-lived owner first.
  """

  kind: str
  parent: Scope | None = None
  owned: list[Var] = field(default_factory=list)
  depth: int = 0

  def __post_init__(self) -> None:
    self.depth = 0 if self.parent is None else self.parent.depth + 1

  def encloses(self, other: Scope | None) -> bool:
    """True if ``other`` is this scope or nested inside it."""
    while other is not None:
      if other is self:
        return True
      other = other.parent
    return False


class Stmt:
  __slots__ = ()


@dataclass(eq=False, slots=True)
class Assign(Stmt):
  var: Var
  value: Expr


@dataclass(eq=False, slots=True)
class Store(Stmt):
  buf: Var
  index: Expr
  value: Expr


@dataclass(eq=False, slots=True)
class If(Stmt):
  cond: Expr
  then: list[Stmt]
  orelse: list[Stmt]


@dataclass(eq=False, slots=True)
class ForRange(Stmt):
  var: Var
  start: Expr
  stop: Expr
  step: Expr
  body: list[Stmt]
  scope: Scope
  simd: bool = False  # iterations are independent: safe to vectorise
  reduction: tuple | None = None  # (op, accumulator): an associative fold, may be reordered


@dataclass(eq=False, slots=True)
class While(Stmt):
  """``while (true) body`` - exits through :class:`Break`."""

  body: list[Stmt]
  scope: Scope


@dataclass(eq=False, slots=True)
class Break(Stmt):
  pass


@dataclass(eq=False, slots=True)
class Continue(Stmt):
  pass


@dataclass(eq=False, slots=True)
class Alloc(Stmt):
  var: Var
  count: Expr
  zero: bool = False


@dataclass(eq=False, slots=True)
class Incref(Stmt):
  buf: Expr


@dataclass(eq=False, slots=True)
class Release(Stmt):
  """Drop the reference held in ``var`` (if any) and clear it."""

  var: Var


@dataclass(eq=False, slots=True)
class ReleaseScope(Stmt):
  """Release everything ``scope`` owns (expanded once the scope is complete)."""

  scope: Scope


@dataclass(eq=False, slots=True)
class ParLoop(Stmt):
  """Run ``body`` for every chunk of ``[0, total)``, possibly concurrently.

  Chunk ``c`` covers ``[c * size, min((c + 1) * size, total))`` where
  ``size = ceil(total / nchunks)``.  The body sees ``chunk``, ``start`` and
  ``stop`` and may only write to buffers (never to outer variables).
  """

  total: Expr
  nchunks: Expr
  chunk: Var
  start: Var
  stop: Var
  body: list[Stmt]
  scope: Scope
  kind: str = "map"  # "map" | "reduce"
  loc: SourceLocation | None = None
  heavy: bool = False  # each iteration contains loops of its own


@dataclass(eq=False, slots=True)
class Check(Stmt):
  """Fail with ``err`` unless ``cond`` holds."""

  cond: Expr
  err: ErrorInfo
  args: tuple[Expr, ...] = ()


@dataclass(eq=False, slots=True)
class Comment(Stmt):
  text: str


# --------------------------------------------------------------------------
# function-level containers
# --------------------------------------------------------------------------


@dataclass(eq=False)
class ArrayParam:
  """How one array crosses the Python/native boundary."""

  buf: Var
  offset: Var
  shape: tuple[Var, ...]
  strides: tuple[Var, ...] | None  # None: C-contiguous, strides derived from shape
  writes: bool = False  # generated code may store into it


@dataclass(eq=False)
class ArrayResult:
  buf: Expr
  offset: Expr
  shape: tuple[Expr, ...]
  strides: tuple[Expr, ...]
  elt: ScalarType


@dataclass(eq=False)
class Function:
  name: str
  params: list[Var | ArrayParam]
  body: list[Stmt]
  results: list[Expr | ArrayResult]
  scope: Scope
  loc: SourceLocation | None = None


# --------------------------------------------------------------------------
# traversal helpers
# --------------------------------------------------------------------------


def sub_blocks(s: Stmt) -> tuple[list[Stmt], ...]:
  if isinstance(s, If):
    return (s.then, s.orelse)
  if isinstance(s, (ForRange, While, ParLoop)):
    return (s.body,)
  return ()


def stmt_exprs(s: Stmt) -> tuple[Expr, ...]:
  """Expressions *read* directly by ``s`` (not by nested blocks)."""
  if isinstance(s, Assign):
    return (s.value,)
  if isinstance(s, Store):
    return (s.buf, s.index, s.value)
  if isinstance(s, If):
    return (s.cond,)
  if isinstance(s, ForRange):
    return (s.start, s.stop, s.step)
  if isinstance(s, Alloc):
    return (s.count,)
  if isinstance(s, Incref):
    return (s.buf,)
  if isinstance(s, ParLoop):
    return (s.total, s.nchunks)
  if isinstance(s, Check):
    return (s.cond, *s.args)
  return ()


def stmt_defs(s: Stmt) -> tuple[Var, ...]:
  """Variables written directly by ``s``."""
  if isinstance(s, Assign):
    return (s.var,)
  if isinstance(s, ForRange):
    return (s.var,)
  if isinstance(s, Alloc):
    return (s.var,)
  if isinstance(s, Release):
    return (s.var,)
  if isinstance(s, ParLoop):
    return (s.chunk, s.start, s.stop)
  return ()


def walk_stmts(block: list[Stmt]) -> Iterator[Stmt]:
  for s in block:
    yield s
    for b in sub_blocks(s):
      yield from walk_stmts(b)


# --------------------------------------------------------------------------
# pretty printer (for ``Specialization.ir`` / debugging)
# --------------------------------------------------------------------------

_INFIX = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "//", "mod": "%",
    "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    "and": "and", "or": "or", "bitand": "&", "bitor": "|", "bitxor": "^",
    "lshift": "<<", "rshift": ">>", "pow": "**",
}  # fmt: skip


def fmt_expr(e: Expr) -> str:
  if isinstance(e, Const):
    return repr(e.value)
  if isinstance(e, Var):
    return e.name
  if isinstance(e, Prim):
    if e.op in _INFIX and len(e.args) == 2:
      return f"({fmt_expr(e.args[0])} {_INFIX[e.op]} {fmt_expr(e.args[1])})"
    return f"{e.op}({', '.join(map(fmt_expr, e.args))})"
  if isinstance(e, Cast):
    return f"{e.type.name}({fmt_expr(e.arg)})"
  if isinstance(e, Load):
    return f"{e.buf.name}[{fmt_expr(e.index)}]"
  if isinstance(e, RtCall):
    return f"@{e.name}({', '.join(map(fmt_expr, e.args))})"
  if isinstance(e, Null):
    return "NULL"
  return repr(e)


def fmt_block(block: list[Stmt], indent: int = 1) -> list[str]:
  pad = "    " * indent
  out: list[str] = []
  for s in block:
    if isinstance(s, Assign):
      out.append(f"{pad}{s.var.name}: {s.var.type} = {fmt_expr(s.value)}")
    elif isinstance(s, Store):
      out.append(f"{pad}{s.buf.name}[{fmt_expr(s.index)}] = {fmt_expr(s.value)}")
    elif isinstance(s, If):
      out.append(f"{pad}if {fmt_expr(s.cond)}:")
      out += fmt_block(s.then, indent + 1) or [f"{pad}    pass"]
      if s.orelse:
        out.append(f"{pad}else:")
        out += fmt_block(s.orelse, indent + 1)
    elif isinstance(s, ForRange):
      tag = "  # simd" if s.simd else ""
      out.append(f"{pad}for {s.var.name} in range({fmt_expr(s.start)}, {fmt_expr(s.stop)}, {fmt_expr(s.step)}):{tag}")
      out += fmt_block(s.body, indent + 1) or [f"{pad}    pass"]
    elif isinstance(s, While):
      out.append(f"{pad}while True:")
      out += fmt_block(s.body, indent + 1) or [f"{pad}    pass"]
    elif isinstance(s, Break):
      out.append(f"{pad}break")
    elif isinstance(s, Continue):
      out.append(f"{pad}continue")
    elif isinstance(s, Alloc):
      z = ", zeroed" if s.zero else ""
      out.append(f"{pad}{s.var.name} = alloc({s.var.type.elt}, {fmt_expr(s.count)}{z})")
    elif isinstance(s, Incref):
      out.append(f"{pad}incref({fmt_expr(s.buf)})")
    elif isinstance(s, Release):
      out.append(f"{pad}release({s.var.name})")
    elif isinstance(s, ReleaseScope):
      if s.scope.owned:
        out.append(f"{pad}release({', '.join(v.name for v in s.scope.owned)})")
    elif isinstance(s, ParLoop):
      out.append(
        f"{pad}parallel[{s.kind}] for {s.chunk.name}, ({s.start.name}, {s.stop.name}) "
        f"in chunks({fmt_expr(s.total)}, n={fmt_expr(s.nchunks)}):"
      )
      out += fmt_block(s.body, indent + 1) or [f"{pad}    pass"]
    elif isinstance(s, Check):
      out.append(f"{pad}check {fmt_expr(s.cond)}  # else {s.err.exc}: {s.err.fmt}")
    elif isinstance(s, Comment):
      out.append(f"{pad}# {s.text}")
    else:  # pragma: no cover
      out.append(f"{pad}{s!r}")
  return out


def fmt_function(fn: Function) -> str:
  params = []
  for p in fn.params:
    if isinstance(p, ArrayParam):
      params.append(f"{p.buf.name}: array[{p.buf.type.elt}, {len(p.shape)}d]")
    else:
      params.append(f"{p.name}: {p.type}")
  lines = [f"def {fn.name}({', '.join(params)}):"]
  lines += fmt_block(fn.body)
  res = []
  for r in fn.results:
    if isinstance(r, ArrayResult):
      res.append(f"array({fmt_expr(r.buf)}, shape=({', '.join(map(fmt_expr, r.shape))}))")
    else:
      res.append(fmt_expr(r))
  lines.append(f"    return {', '.join(res) if res else 'None'}")
  return "\n".join(lines)


__all__ = [n for n in dir() if n[0].isupper()] + [
    "children", "walk_expr", "walk_stmts", "sub_blocks", "stmt_exprs", "stmt_defs",
    "is_trivial", "same", "fmt_expr", "fmt_block", "fmt_function",
]  # fmt: skip
_ = (Bool, Int64)
