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

from ..errors import SourceLocation
from ..ndtypes import Bool, BufType, Int64, ScalarType

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
  atomic: str | None = None


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
  reduction: tuple | None = None  # (op, accumulator or accumulators): associative folds, may be reordered


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
  work: Expr | None = None  # estimated inner iterations per element (None: one), valid just before the loop
  shared_tile: SharedTilePlan | None = None  # optional GPU __shared__/threadgroup block-tiling plan


# ---- operators -------------------------------------------------------------
#
# The frontend does not emit the loops of an array operation itself.  It
# emits one of the statements below, whose body is a block of this same IR
# parameterised by index variables; the passes in parakeet.transforms rewrite
# them (fusion, tiling) and the last one lowers them to the loops above.


@dataclass(eq=False, slots=True)
class IndexMap(Stmt):
  """Run ``body`` once for every index of ``shape``.

  Unless ``ordered``, the iterations are independent: they may run in any
  order or at the same time, and the body writes nothing but array elements
  no other iteration touches."""

  shape: tuple[Expr, ...]
  index: tuple[Var, ...]
  body: list[Stmt]
  scope: Scope
  ordered: bool = False
  parallel: bool = True  # may use threads or a device (false: nested in a parallel loop, or tiny)
  loc: SourceLocation | None = None


@dataclass(eq=False, slots=True)
class Payload:
  """One auxiliary value carried alongside a :class:`Fold` accumulator (e.g.
  the winning index in ``argmin``/``argmax``, or running counts/sums in online
  variance / online softmax)."""

  result: Var
  value: Expr
  a: Var
  b: Var
  combined: Expr


@dataclass(eq=False, slots=True)
class Fold:
  """One accumulator of an :class:`IndexReduce`: ``result = fold(combine, value(idx))``.

  ``combine`` is the block that computes ``combined`` from the accumulator
  ``a`` and the next value ``b``; ``op`` names it when it is an associative
  operation (the fold may then be regrouped across chunks and, for ``add``/``mul``,
  vectorised).  ``payloads`` carries zero, one, or many auxiliary values updated
  alongside ``a`` by the same combiner."""

  result: Var
  value: Expr
  a: Var
  b: Var
  combine: list[Stmt]
  combined: Expr
  init: Expr | None = None
  op: str | None = None
  empty: ErrorInfo | None = None  # raised for an empty index space when there is no init
  payloads: list[Payload] = field(default_factory=list)


@dataclass(eq=False, slots=True)
class IndexReduce(Stmt):
  """Fold the values ``body`` computes at every index of ``shape`` (in index
  order, unless every fold's ``op`` allows regrouping).

  With ``where`` it is the thesis's *FilterReduce*: ``pre`` runs at every
  index, and only where ``where`` then holds does ``body`` run and its value
  get folded.  Such a reduction always has an ``init``."""

  shape: tuple[Expr, ...]
  index: tuple[Var, ...]
  body: list[Stmt]
  folds: list[Fold]
  scope: Scope
  home: Scope  # owns the temporaries lowering needs
  parallel: bool = False
  loc: SourceLocation | None = None
  pre: list[Stmt] = field(default_factory=list)
  where: Expr | None = None


@dataclass(eq=False, slots=True)
class IndexFilter(Stmt):
  """The indices in ``[0, n)`` at which ``keep`` holds, in order: allocates
  the int64 buffer ``out`` and sets ``count`` to how many there are."""

  n: Expr
  index: Var
  body: list[Stmt]
  keep: Expr
  out: Var
  count: Var
  scope: Scope
  parallel: bool = False
  loc: SourceLocation | None = None


@dataclass(eq=False, slots=True)
class IndexScan(Stmt):
  """A running fold over ``[0, n)``: at every index, in order, the value
  ``body`` computes is folded in, and ``emit`` runs with ``fold.result``
  holding the running value (it usually stores it).  ``parallel`` says the
  combiner is associative, so chunks may be folded independently first."""

  n: Expr
  index: Var
  body: list[Stmt]
  fold: Fold
  emit: list[Stmt]
  scope: Scope
  parallel: bool = False
  loc: SourceLocation | None = None


@dataclass(eq=False, slots=True)
class TileStage:
  """An N-dimensional tile of an input expression (or fused prologue) staged
  into a tile-local buffer before an :class:`IndexTile`'s body runs.

  Works for any number of inputs, any rank, and any axis traversal: ``body``
  computes ``value`` at tile-local coordinates ``index`` over ``shape`` and
  writes into ``buf`` (L1 scratch on CPU, ``__shared__`` / ``threadgroup``
  memory on GPUs)."""

  buf: Var
  shape: tuple[Expr, ...]
  index: tuple[Var, ...]
  body: list[Stmt]
  value: Expr
  scope: Scope


@dataclass(eq=False)
class SharedTilePlan:
  """GPU ``__shared__`` / ``threadgroup`` block-tiling plan for an N-D
  map-of-reduction (any batch rank, any number of staged prologue inputs,
  arbitrary folds/payloads, and arbitrary epilogue).

  Attached to an outer :class:`IndexTile` by :mod:`parakeet.transforms.tiling`
  and preserved onto its lowered :class:`ParLoop` so CUDA and Metal backends
  emit cooperative shared-memory + register-blocked kernels while the CPU
  runs ``ParLoop.body``."""

  shape: tuple[Expr, ...]  # (batch..., M, N)
  index: tuple[Var, ...]  # (b..., i, j)
  before: list[Stmt]  # original map setup statements (for epilogue / mixed deps)
  reduce: IndexReduce  # original inner reduction (1D extent K)
  after: list[Stmt]  # epilogue statements (activation, bias, output stores)
  stages: list[TileStage]  # staged operands/prologues
  stage_sides: tuple[str, ...]  # parallel to stages: "row" | "col" | "k"
  stage_vars: tuple[Var, ...]  # scalar placeholder Vars in mixed_body/mixed_folds
  mixed_body: list[Stmt]  # reduction body with staged expressions replaced by stage_vars
  mixed_folds: list[Fold]  # reduction folds with staged expressions replaced by stage_vars
  has_mixed_loads: bool = False  # True if mixed_body/mixed_folds still contain global Loads
  pre_checks: list[Check] = field(default_factory=list)  # uniform (i,j-invariant) checks from before


@dataclass(eq=False, slots=True)
class IndexTile(Stmt):
  """Split an N-dimensional index space ``shape`` into tiles of ``tile_shape``.

  Along each axis ``k``:

  * if ``tile_shape[k]`` is an expression ``B``, ``origin[k]`` steps by ``B``
    across ``[0, shape[k])`` and ``extent[k] = min(B, shape[k] - origin[k])``;
  * if ``tile_shape[k] is None``, axis ``k`` is left whole (``origin[k] = 0``,
    ``extent[k] = shape[k]``).

  ``body`` holds ordinary operator IR (including nested :class:`IndexTile`s),
  so multi-level cache blocking, K-tiling, reduction interchange, register
  jamming, and GPU shared-memory tiling are all compositions of one operator.
  """

  shape: tuple[Expr, ...]
  tile_shape: tuple[Expr | None, ...]
  origin: tuple[Var, ...]
  extent: tuple[Var, ...]
  stages: list[TileStage]
  body: list[Stmt]
  scope: Scope
  level: str = "cache"  # "cache" | "shared" | "register"
  parallel: bool = True
  loc: SourceLocation | None = None
  shared_tile: SharedTilePlan | None = None


@dataclass(eq=False, slots=True)
class IndexScatter(Stmt):
  """Generalized scatter-reduce (``ReduceByIndex`` / histogram / scatter-add):
  for every coordinate ``index`` in ``shape``, evaluate ``body`` and fold
  ``value`` into ``buf[target]`` using associative ``op`` (``"add"``, ``"mul"``,
  ``"min"``, ``"max"``, or ``"assign"``), optionally guarded by ``where``.

  Lowers to thread-privatized partial buffers combined into ``buf`` when
  ``parallel`` and ``dest_size`` fits the privatization budget (works cleanly
  on multi-core CPU, CUDA, and Metal without requiring float/int64 atomics),
  falling back to a sequential loop otherwise."""

  shape: tuple[Expr, ...]
  index: tuple[Var, ...]
  body: list[Stmt]
  buf: Var
  dest_size: Expr
  target: Expr
  value: Expr
  op: str  # "add" | "mul" | "min" | "max" | "assign"
  scope: Scope
  parallel: bool = True
  loc: SourceLocation | None = None
  where: Expr | None = None


OPERATORS = (IndexMap, IndexReduce, IndexFilter, IndexScan, IndexTile, IndexScatter)


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
  if isinstance(s, (ForRange, While, ParLoop, IndexMap, IndexFilter, IndexScatter)):
    return (s.body,)
  if isinstance(s, IndexTile):
    return (*[st.body for st in s.stages], s.body)
  if isinstance(s, IndexReduce):
    return (s.pre, s.body, *[f.combine for f in s.folds])
  if isinstance(s, IndexScan):
    return (s.body, s.fold.combine, s.emit)
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
  if isinstance(s, IndexMap):
    return s.shape
  if isinstance(s, IndexScatter):
    return (*s.shape, s.buf, s.dest_size, s.target, s.value, *(() if s.where is None else (s.where,)))
  if isinstance(s, IndexTile):
    out: list[Expr] = list(s.shape) + [d for d in s.tile_shape if d is not None]
    for st in s.stages:
      out.extend(st.shape)
      out.append(st.value)
    return tuple(out)
  if isinstance(s, IndexFilter):
    return (s.n, s.keep)
  if isinstance(s, IndexScan):
    f = s.fold
    return (s.n, f.value, f.combined) + ((f.init,) if f.init is not None else ())
  if isinstance(s, IndexReduce):
    out = list(s.shape) + ([s.where] if s.where is not None else [])
    for f in s.folds:
      out += [f.value, f.combined] + ([f.init] if f.init is not None else [])
      for p in f.payloads:
        out += [p.value, p.combined]
    return tuple(out)
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
  if isinstance(s, IndexMap):
    return s.index
  if isinstance(s, IndexScatter):
    return s.index
  if isinstance(s, IndexTile):
    defs: list[Var] = [*s.origin, *s.extent]
    for st in s.stages:
      defs.append(st.buf)
      defs.extend(st.index)
    return tuple(defs)
  if isinstance(s, IndexFilter):
    return (s.index, s.out, s.count)
  if isinstance(s, IndexScan):
    return (s.index, s.fold.result, s.fold.a, s.fold.b)
  if isinstance(s, IndexReduce):
    defs = list(s.index)
    for f in s.folds:
      defs.extend((f.result, f.a, f.b))
      for p in f.payloads:
        defs.extend((p.result, p.a, p.b))
    return tuple(defs)
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
      op = f" {s.atomic}= /*atomic*/ " if s.atomic else " = "
      out.append(f"{pad}{s.buf.name}[{fmt_expr(s.index)}]{op}{fmt_expr(s.value)}")
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
    elif isinstance(s, IndexMap):
      how = "ordered " if s.ordered else ("" if s.parallel else "nested ")
      out.append(f"{pad}{how}map {', '.join(v.name for v in s.index)} over ({', '.join(map(fmt_expr, s.shape))}):")
      out += fmt_block(s.body, indent + 1) or [f"{pad}    pass"]
    elif isinstance(s, IndexTile):
      how = "parallel " if s.parallel else ""
      tiles = ", ".join("*" if d is None else fmt_expr(d) for d in s.tile_shape)
      coords = ", ".join(f"({o.name}:{e.name})" for o, e in zip(s.origin, s.extent))
      out.append(f"{pad}{how}tile[{s.level}] {coords} over ({', '.join(map(fmt_expr, s.shape))}) by ({tiles}):")
      for st in s.stages:
        out.append(
          f"{pad}    stage {st.buf.name}[{', '.join(v.name for v in st.index)}] over ({', '.join(map(fmt_expr, st.shape))}):"
        )
        out += fmt_block(st.body, indent + 2)
        out.append(f"{pad}        = {fmt_expr(st.value)}")
      out += fmt_block(s.body, indent + 1) or [f"{pad}    pass"]
    elif isinstance(s, IndexReduce):
      how = "parallel " if s.parallel else ""
      out.append(f"{pad}{how}reduce {', '.join(v.name for v in s.index)} over ({', '.join(map(fmt_expr, s.shape))}):")
      out += fmt_block(s.pre, indent + 1)
      if s.where is not None:
        out.append(f"{pad}    where {fmt_expr(s.where)}:")
      out += fmt_block(s.body, indent + (1 if s.where is None else 2))
      for f in s.folds:
        init = "" if f.init is None else f", init={fmt_expr(f.init)}"
        if f.payloads:
          lhs = ", ".join([f.result.name, *(p.result.name for p in f.payloads)])
          vals = ", ".join([fmt_expr(f.value), *(fmt_expr(p.value) for p in f.payloads)])
          how = f.op or f"lambda ({f.a.name}, ...), ({f.b.name}, ...): ({fmt_expr(f.combined)}, ...)"
          out.append(f"{pad}    {lhs} = fold({how}, ({vals}){init})")
        else:
          if f.op is not None and not f.combine:
            how = f.op
          else:
            how = f"lambda {f.a.name}, {f.b.name}: {fmt_expr(f.combined)}"
          out.append(f"{pad}    {f.result.name}: {f.result.type} = fold({how}, {fmt_expr(f.value)}{init})")
        if f.op is None or f.combine:
          out += fmt_block(f.combine, indent + 2)
    elif isinstance(s, IndexScan):
      how, f = "parallel " if s.parallel else "", s.fold
      out.append(f"{pad}{how}scan {s.index.name} over ({fmt_expr(s.n)}):")
      out += fmt_block(s.body, indent + 1)
      init = "" if f.init is None else f", init={fmt_expr(f.init)}"
      out.append(
        f"{pad}    {f.result.name}: {f.result.type} = "
        f"fold(lambda {f.a.name}, {f.b.name}: {fmt_expr(f.combined)}, {fmt_expr(f.value)}{init})"
      )
      out += fmt_block(f.combine, indent + 2)
      out += fmt_block(s.emit, indent + 1)
    elif isinstance(s, IndexFilter):
      how = "parallel " if s.parallel else ""
      out.append(f"{pad}{how}filter {s.index.name} over ({fmt_expr(s.n)}):")
      out += fmt_block(s.body, indent + 1)
      out.append(f"{pad}    {s.out.name}, {s.count.name} = keep({s.index.name}, where={fmt_expr(s.keep)})")
    elif isinstance(s, IndexScatter):
      how = "parallel " if s.parallel else ""
      out.append(
        f"{pad}{how}scatter[{s.op}] {', '.join(v.name for v in s.index)} over ({', '.join(map(fmt_expr, s.shape))}):"
      )
      out += fmt_block(s.body, indent + 1)
      guard = "" if s.where is None else f" if {fmt_expr(s.where)}"
      out.append(f"{pad}    {s.buf.name}[{fmt_expr(s.target)}] {s.op}= {fmt_expr(s.value)}{guard}")
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
