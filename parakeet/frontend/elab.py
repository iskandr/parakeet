"""The frontend: turn a Python function plus argument types into typed loop IR.

The :class:`Elaborator` is a *staged interpreter*.  It walks the Python AST of
the function with an environment of compile-time values (see
:mod:`parakeet.syntax.values`) and emits IR for whatever can only be known at run
time.  Calls are inlined, closures and tuples disappear, and array expressions
are fused (see :mod:`parakeet.frontend.arrays`).

Three things cannot be decided in a single forward pass, and are handled by
*restarting* elaboration with better assumptions recorded in
:class:`ElabState`:

* the type of a variable modified in a loop (widened until stable),
* whether an array expression must be evaluated eagerly,
* which scalar arguments must be compile-time constants (e.g. ``axis``).
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import enum
import mmap
import types as pytypes
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np

from .. import prims
from ..errors import (
  CompileError,
  InternalError,
  ParakeetError,
  SourceLocation,
  StaticValueError,
  TypeInferenceError,
  UnsupportedError,
)
from ..ndtypes import Bool, BufType, ScalarType, WeakBool, WeakFloat, WeakInt, join
from ..ragged import RaggedArray
from ..syntax import ops
from ..syntax.ir import (
  ArrayParam,
  ArrayResult,
  Assign,
  Break,
  Const,
  Continue,
  Expr,
  ForRange,
  Function,
  If,
  Incref,
  IndexScatter,
  Load,
  Null,
  Release,
  ReleaseScope,
  Scope,
  Store,
  Var,
  While,
  is_trivial,
  walk_expr,
  walk_stmts,
)
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, imul, isub
from ..syntax.records import Record, StructuredRow, StructuredView, record_type
from ..syntax.values import (
  ArrayVal,
  BoundMethod,
  Closure,
  EnumerateVal,
  Partial,
  PyFunc,
  RangeVal,
  Root,
  TypeVal,
  Undefined,
  ZipVal,
  describe,
)
from ..target import TargetProfile
from ..transforms import autodiff as _autodiff
from .arrays import ArrayMixin, LiveRoots, Restart
from .signature import CODE_FUNCTIONS, STAGED, Key, Ref, Unsupported, classify, type_value, with_static_leaves
from .source import FunctionSource, assigned_names, get_source

MAX_INLINE_DEPTH = 100
_MISSING = object()


@dataclass
class Options:
  """Frontend switches (part of every specialization key)."""

  bounds_check: bool = True
  parallel: bool = True
  fusion: bool = True
  tiling: bool = True
  tile_bytes: int = 256 * 1024
  blas: bool = False  # hand float matrix products to the BLAS NumPy uses
  skip: tuple = ()  # middle-end passes not to run
  target: TargetProfile = field(default_factory=TargetProfile)

  def key(self) -> tuple:
    return (
      self.bounds_check,
      self.parallel,
      self.fusion,
      self.tiling,
      self.tile_bytes,
      self.blas,
      self.skip,
      self.target,
    )


@dataclass
class ElabState:
  """Assumptions that survive restarts of one specialization."""

  eager: set[int] = field(default_factory=set)
  strict: set[int] = field(default_factory=set)
  keep: list = field(default_factory=list)  # keeps AST nodes alive so ids stay unique
  hints: dict = field(default_factory=dict)
  static_leaves: set[int] = field(default_factory=set)
  static_implicit: set = field(default_factory=set)
  tokens: dict = field(default_factory=dict)  # stand-in origins for arrays made by the backward pass
  restarts: int = 0


@dataclass(eq=False)
class Implicit:
  """A global/closure value read by compiled code, re-fetched on every call."""

  key: tuple
  getter: Any
  type_key: Key
  static: frozenset[int]


@dataclass(eq=False)
class Ret:
  value: Any = _MISSING
  target: Any = _MISSING
  vtype: Any = None
  flag: Var | None = None
  done: bool = False


@dataclass(eq=False)
class LoopCtx:
  node: ast.AST
  home: Scope
  pre_block: list
  scope: Scope
  carried: dict = field(default_factory=dict)
  post: dict = field(default_factory=dict)
  post_names: tuple = ()
  static: dict = field(default_factory=dict)
  ret_seen: bool = False
  exhausted: Var | None = None


@dataclass(eq=False)
class Frame:
  src: FunctionSource
  node: ast.AST
  env: dict
  lexical: Frame | None
  cells: dict | None
  globals: dict
  path: tuple
  scope: Scope
  entry_block: list
  entry_pos: int
  call_loc: SourceLocation | None
  locals: frozenset
  ret: Ret = field(default_factory=Ret)
  loops: list = field(default_factory=list)


@dataclass(eq=False)
class TargetInfo:
  home: Scope
  hint_key: Any
  slot: Var | None = None


@dataclass(eq=False)
class Elaborated:
  function: Function
  result_spec: tuple
  implicits: list[Implicit]
  writes_unknown: bool
  operators: str = ""  # the function before the middle-end passes, as text


_BINOPS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "floordiv",
    ast.Mod: "mod", ast.Pow: "pow", ast.BitAnd: "bitand", ast.BitOr: "bitor",
    ast.BitXor: "bitxor", ast.LShift: "lshift", ast.RShift: "rshift",
}  # fmt: skip
_CMPOPS = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
_UNARY = {ast.USub: "neg", ast.UAdd: "pos", ast.Invert: "invert"}
_SYMBOL = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "//", "mod": "%", "pow": "**",
    "bitand": "&", "bitor": "|", "bitxor": "^", "lshift": "<<", "rshift": ">>", "eq": "==",
    "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "neg": "-", "pos": "+", "invert": "~",
}  # fmt: skip
_EXCEPTIONS = {
  n: getattr(builtins, n)
  for n in (
    "ValueError TypeError IndexError ZeroDivisionError AssertionError RuntimeError "
    "ArithmeticError OverflowError KeyError NotImplementedError FloatingPointError"
  ).split()
}


def _sanitize(name: str) -> str:
  out = "".join(c if (c.isascii() and (c.isalnum() or c == "_")) else "_" for c in name)
  return out if out and not out[0].isdigit() else "v" + out


class Elaborator(ArrayMixin):
  def __init__(self, fn: pytypes.FunctionType, arg_keys: tuple, state: ElabState, options: Options):
    self.state = state
    self.options = options
    self.root_fn = fn
    self.arg_keys = arg_keys
    self.fn_scope = Scope("function")
    self.scope = self.fn_scope
    self.entry_block: list = []
    self.body_block: list = []
    self.block = self.body_block
    self.loop_depth = 0
    self.par_depth = 0
    self.frames: list[Frame] = []
    self.node_stack: list[ast.AST] = []
    self.params: list = []
    self.param_origin: dict[int, tuple] = {}
    self.implicits: list[Implicit] = []
    self.implicit_cache: dict = {}
    self.array_params: list[ArrayParam] = []
    self.mutables: set[int] = set()
    self.nonneg: set[int] = set()
    self.not_one: set[int] = set()
    self.facts: dict[int, int] = {}
    self.upper: dict[int, Expr] = {}
    self.stores: list = []
    self.writes_unknown = False
    self.journal: list | None = None
    self.targets: dict[int, TargetInfo] = {}
    self._keep: list = []
    self.results: list = []
    self.pure_defs: dict[int, Expr] = {}
    self.jam = None  # set while a tiled map collects its elements' reductions
    # reverse-mode & forward-mode differentiation (autodiff / jvp)
    self.tape = None  # records array operators applied to differentiated values
    self.ad_live = 0  # > 0 between the start of a grad() and the end of its backward pass
    self.jvp = None  # active forward-mode JVPContext (parakeet.transforms.jvp)
    self.jvp_live = 0  # > 0 between the start of a jvp() and its completion
    self.ad_vars: set[int] = set()  # float variables that depend on a differentiated value
    self.ad_bufs: set[int] = set()  # buffers holding such values
    self.tainted_vars: dict[int, tuple] = {}  # ... that the tape cannot see through
    self.tainted_bufs: dict[int, tuple] = {}
    self.defs: dict[int, Expr] = {}  # pure definitions of float variables
    self.if_stack: list = []  # (frame, node) of the if statements with a run-time condition being elaborated
    self.both_depth = 0  # > 0 inside an if statement whose branches are both being run
    self.reach: dict[int, set | None] = {}  # buffer of a merged array -> the allocations it may be
    self.lib_calls: list[tuple[str, int]] = []  # library functions being elaborated (name, frame depth)

  # ================================================================ plumbing

  @property
  def frame(self) -> Frame:
    return self.frames[-1]

  def emit(self, stmt) -> None:
    self.block.append(stmt)
    if self.tape is not None:
      _autodiff.observe(self, stmt)

  def tmp(self, e, name: str = "t") -> Expr:
    """Bind a non-trivial scalar expression to a fresh variable."""
    if isinstance(e, int):
      return Const(e, I)
    if is_trivial(e):
      return e
    v = Var(name, e.type)
    self.block.append(Assign(v, e))
    if e.type.kind in "iu":
      if self.is_nonneg(e):
        self.nonneg.add(v.uid)
      self.pure_defs[v.uid] = e
    elif self.ad_live or self.jvp_live:
      if self.ad_live:
        _autodiff.note_def(self, v, e)
      elif e.type.kind == "f":
        self.defs[v.uid] = e
    return v

  def copy_scalar(self, e: Expr, name: str = "t") -> Expr:
    if isinstance(e, Const):
      return e
    v = Var(name, e.type)
    self.emit(Assign(v, e))
    jvp = self.jvp
    if jvp is not None and not jvp.muted and e.type.kind == "f":
      dt = jvp.tangent_expr(self, e)
      if dt is not None:
        dv = Var(f"d_{name}", v.type)
        with jvp.computing(self):
          self.emit(Assign(dv, cast(dt, v.type)))
        jvp.scalar_t[v.uid] = dv
    return v

  def save_facts(self):
    return dict(self.facts), set(self.nonneg), set(self.not_one)

  def restore_facts(self, saved) -> None:
    self.facts, self.nonneg, self.not_one = dict(saved[0]), set(saved[1]), set(saved[2])

  @contextlib.contextmanager
  def in_block(self, blk: list, scope: Scope | None = None, loop: bool = False, par: bool = False):
    saved = (self.block, self.scope, self.save_facts())
    self.block = blk
    if scope is not None:
      self.scope = scope
    self.loop_depth += loop
    self.par_depth += par
    try:
      yield
    finally:
      self.loop_depth -= loop
      self.par_depth -= par
      self.block, self.scope = saved[0], saved[1]
      self.restore_facts(saved[2])

  @contextlib.contextmanager
  def trial(self):
    """Elaborate into a scratch block whose effects are discarded."""
    blk: list = []
    owned_len = len(self.scope.owned)
    stores_len = len(self.stores)
    journal, self.journal = self.journal, []
    writes = self.writes_unknown
    param_writes = [p.writes for p in self.array_params]
    n_frames = len(self.frames)
    try:
      with self.in_block(blk):
        yield blk
    finally:
      for arr, uses in reversed(self.journal):
        arr.uses = uses
      self.journal = journal
      del self.scope.owned[owned_len:]
      del self.stores[stores_len:]
      del self.frames[n_frames:]
      self.writes_unknown = writes
      for p, w in zip(self.array_params, param_writes):
        p.writes = w

  def capture(self):
    """Snapshot the lexical context so user code can be elaborated later
    (when a delayed array is finally evaluated) exactly as it would have
    been now."""
    return (list(self.frames), [dict(f.env) for f in self.frames], list(self.node_stack))

  @contextlib.contextmanager
  def restored(self, ctx):
    frames, envs, nodes = ctx
    saved_frames, saved_nodes = self.frames, self.node_stack
    saved_envs = [f.env for f in frames]
    self.frames, self.node_stack = list(frames), list(nodes)
    for f, e in zip(frames, envs):
      f.env = dict(e)
    try:
      yield
    finally:
      for f, e in zip(frames, saved_envs):
        f.env = e
      self.frames, self.node_stack = saved_frames, saved_nodes

  def loc(self, node: ast.AST | None = None) -> SourceLocation | None:
    if not self.frames:
      return None
    if node is None:
      node = self.node_stack[-1] if self.node_stack else None
    fr = self.frame
    if node is not None and not hasattr(node, "lineno"):
      node = None
    return fr.src.location(node)

  def call_chain(self) -> tuple[SourceLocation, ...]:
    return tuple(f.call_loc for f in self.frames if f.call_loc is not None)

  def error(self, cls, message: str, node: ast.AST | None = None, hint: str | None = None) -> CompileError:
    return cls(message, self.loc(node), self.call_chain(), hint)

  def hint_get(self, key):
    return self.state.hints.get(key)

  def restart(self, why: str) -> None:
    raise Restart(why)

  # ============================================================== value types

  def vtype(self, v) -> tuple:
    """Hashable type descriptor of a compile-time value."""
    if isinstance(v, Expr):
      return ("s", v.type)
    if isinstance(v, ArrayVal):
      return ("a", v.elt, v.rank)
    if record := record_type(v):
      return ("rec", record, self.vtype(record.values(v)))
    if isinstance(v, tuple):
      return ("t", *[self.vtype(x) for x in v])
    if v is None:
      return ("n",)
    if isinstance(v, TypeVal):
      return ("ty", v.type)
    if isinstance(v, Closure):
      return ("cl", id(v.node))
    if isinstance(v, PyFunc):
      return ("fn", v.fn.__code__)
    if isinstance(v, str):
      return ("str", v)
    if isinstance(v, RangeVal):
      return ("r",)
    if isinstance(v, slice):
      return ("sl", self.vtype(v.start), self.vtype(v.stop), self.vtype(v.step))
    if isinstance(v, BoundMethod):
      return ("bm", v.name, self.vtype(v.obj))
    if isinstance(v, Undefined):
      return ("undef",)
    return ("o", id(v))

  def join_vtypes(self, a, b):
    if a is None:
      return b
    if b is None:
      return a
    if a == b:
      return a
    if a[0] != b[0]:
      return None
    if a[0] == "s":
      return ("s", join(a[1], b[1]))
    if a[0] == "a":
      if a[2] != b[2]:
        return None
      return ("a", join(a[1], b[1]).strong, a[2])
    if a[0] == "rec":
      parts = self.join_vtypes(a[2], b[2]) if a[1] == b[1] else None
      return ("rec", a[1], parts) if parts is not None else None
    if a[0] in ("t", "sl"):
      if len(a) != len(b):
        return None
      parts = [self.join_vtypes(x, y) for x, y in zip(a[1:], b[1:])]
      if any(p is None for p in parts):
        return None
      return (a[0], *parts)
    return None

  def describe_vtype(self, t) -> str:
    if t[0] == "s":
      return describe(Var("x", t[1]))
    if t[0] == "a":
      return f"array[{t[1].name}, {t[2]}d]"
    if t[0] == "rec":
      schema, fields = t[1], t[2][1:]
      parts = (f"{name}={self.describe_vtype(field)}" for name, field in zip(schema.names, fields))
      return f"{schema.cls.__name__}({', '.join(parts)})"
    if t[0] == "t":
      return "(" + ", ".join(self.describe_vtype(x) for x in t[1:]) + ")"
    if t[0] == "n":
      return "None"
    return {"cl": "function", "fn": "function", "ty": "dtype", "str": "str", "r": "range"}.get(t[0], "object")

  # ================================================================ targets

  def make_target(
    self,
    vt,
    like,
    name: str,
    home: Scope,
    hint_key=None,
    init_block: list | None = None,
    unknown_roots: bool = False,
  ):
    """A mutable twin of a value of type ``vt`` (for variables that merge)."""
    tag = vt[0]
    name = _sanitize(name)
    if tag == "s":
      var = Var(name, vt[1])
      self.mutables.add(var.uid)
      if init_block is not None:
        init_block.append(Assign(var, Const(0, vt[1])))
      return var
    if tag == "a":
      elt, rank = vt[1], vt[2]
      buf = Var(name, BufType(elt))
      off = Var(name + "_off", I)
      shape = tuple(Var(f"{name}_n{k}", I) for k in range(rank))
      strides = tuple(Var(f"{name}_s{k}", I) for k in range(rank))
      for v in (buf, off, *shape, *strides):
        self.mutables.add(v.uid)
      self.nonneg.update(v.uid for v in shape)
      if init_block is not None:
        init_block.append(Assign(buf, Null(buf.type)))
        for v in (off, *shape, *strides):
          init_block.append(Assign(v, ZERO))
      unknown = unknown_roots or (hint_key is not None and self.hint_get((hint_key, "roots")) == "unknown")
      arr = ArrayVal(
                elt, shape, buf=buf, offset=off, strides=strides,
                roots=None if unknown else LiveRoots((), hint_key),
            )  # fmt: skip
      self.targets[id(arr)] = TargetInfo(home, hint_key)
      self._keep.append(arr)
      return arr
    if tag == "rec":
      record = vt[1]
      return record.build(self.make_target(vt[2], record.values(like), name, home, hint_key, init_block, unknown_roots))
    if tag in ("t", "sl"):
      likes = like if isinstance(like, tuple) else (like.start, like.stop, like.step)
      parts = tuple(
        self.make_target(
          t, x, f"{name}_{k}", home, None if hint_key is None else (hint_key, k), init_block, unknown_roots
        )
        for k, (t, x) in enumerate(zip(vt[1:], likes))
      )
      return parts if tag == "t" else slice(*parts)
    return like  # static values merge only with themselves

  def assign_target(self, target, v) -> None:
    jvp = self.jvp
    if isinstance(target, Var):
      sv = self.as_scalar(v, "value")
      dv = jvp.tangent_expr(self, sv) if (jvp is not None and not jvp.muted and target.type.kind == "f") else None
      self.emit(Assign(target, cast(sv, target.type)))
      if jvp is not None and not jvp.muted and target.uid in jvp.targets:
        dt_var = jvp.targets[target.uid]
        with jvp.computing(self):
          self.emit(Assign(dt_var, cast(dv, dt_var.type) if dv is not None else Const(0.0, dt_var.type)))
      return
    elif isinstance(target, ArrayVal):
      info = self.targets[id(target)]
      c = self.as_concrete(self.cast_array(v, target.elt))
      self.emit(Assign(target.buf, c.buf))
      self.emit(Assign(target.offset, c.offset))
      for tv, sv in zip(target.shape + target.strides, c.shape + c.strides):
        self.emit(Assign(tv, sv))
      roots = self.get_roots(c)
      # which allocations a merged variable may refer to, kept even when its
      # alias information is given up on: a store through it reaches only those
      source = roots if roots is not None else self.reach.get(c.buf.uid)
      mine = self.reach.setdefault(target.buf.uid, set())
      if source is None or mine is None:
        self.reach[target.buf.uid] = None
      else:
        mine |= source
      live = target.roots
      if isinstance(live, LiveRoots):
        if roots is None:
          self._roots_unknown(target, live)
        else:
          new = roots - live.known
          if new:
            if live.consulted:
              self._roots_unknown(target, live)
            else:
              live.known |= new
      if roots is None or any(r.scope is not None and not r.scope.encloses(info.home) for r in roots):
        if info.slot is None:
          info.slot = Var("keep", target.buf.type)
          info.home.owned.append(info.slot)
        self.emit(Incref(c.buf))
        self.emit(Release(info.slot))
        self.emit(Assign(info.slot, c.buf))
      if jvp is not None and not jvp.muted:
        jvp.assign_tangent_target(self, target, c)
      return
    elif record := record_type(target):
      self.assign_target(record.values(target), record.values(v))
    elif isinstance(target, tuple):
      for t, x in zip(target, v):
        self.assign_target(t, x)
    elif isinstance(target, slice):
      for t, x in zip((target.start, target.stop, target.step), (v.start, v.stop, v.step)):
        self.assign_target(t, x)

  def _roots_unknown(self, target: ArrayVal, live: LiveRoots) -> None:
    if live.consulted:
      if live.hint_key is None:
        raise InternalError("alias information of a merged array changed after use")
      self.state.hints[(live.hint_key, "roots")] = "unknown"
      self.restart("array aliasing")
    target.roots = None

  def refs_mutable(self, e: Expr) -> bool:
    return any(isinstance(x, Var) and x.uid in self.mutables for x in walk_expr(e))

  def snapshot(self, v):
    """Copy a value out of mutable variables (for simultaneous assignment)."""
    if isinstance(v, Expr):
      return self.copy_scalar(v) if self.refs_mutable(v) else v
    if isinstance(v, ArrayVal):
      if v.buf is None:
        return self.as_concrete(v)
      comps = (v.buf, v.offset, *v.shape, *v.strides)
      if not any(isinstance(c, Var) and c.uid in self.mutables for c in comps):
        return v
      buf = Var("snap", v.buf.type)
      self.emit(Assign(buf, v.buf))
      return ArrayVal(
                v.elt, tuple(self.copy_scalar(s) for s in v.shape), buf=buf,
                offset=self.copy_scalar(v.offset), strides=tuple(self.copy_scalar(s) for s in v.strides),
                roots=self.get_roots(v), contig=v.contig,
            )  # fmt: skip
    if record := record_type(v):
      return record.build(self.snapshot(x) for x in record.values(v))
    if isinstance(v, tuple):
      return tuple(self.snapshot(x) for x in v)
    return v

  @staticmethod
  def static_equal(a, b) -> bool:
    if a is b:
      return True
    if isinstance(a, Const) and isinstance(b, Const):
      return a.type is b.type and a.value == b.value and type(a.value) is type(b.value)
    if isinstance(a, TypeVal) and isinstance(b, TypeVal):
      return a.type is b.type
    if isinstance(a, str) and isinstance(b, str):
      return a == b
    if record := record_type(a):
      return record == record_type(b) and Elaborator.static_equal(record.values(a), record.values(b))
    if isinstance(a, tuple) and type(a) is type(b):
      return len(a) == len(b) and all(Elaborator.static_equal(x, y) for x, y in zip(a, b))
    return False

  # ============================================================== parameters

  def value_from_key(self, key: Key, source, pos: list[int], name: str = "arg"):
    """Build the compile-time value for a classified argument, creating
    IR parameters for its runtime leaves."""
    tag = key[0]
    name = _sanitize(name)
    if tag == "s":
      v = Var(name, key[1])
      self.params.append(v)
      self.param_origin[v.uid] = (source, pos[0])
      pos[0] += 1
      return v
    if tag == "c":
      pos[0] += 1
      return Const(key[2], key[1])
    if tag == "a":
      leaf = pos[0]
      pos[0] += 1
      elt, rank, layout, ones = key[1], key[2], key[3], key[4]
      fixed = key[5] if len(key) > 5 else (None,) * rank  # dimensions this specialization is keyed on
      buf = Var(name, BufType(elt))
      off = Var(name + "_off", I)
      shape_vars = tuple(Var(f"{name}_n{k}", I) for k in range(rank))
      self.nonneg.update(v.uid for v in shape_vars)
      self.nonneg.add(off.uid)
      shape: list[Expr] = []
      for k, (v, one) in enumerate(zip(shape_vars, ones)):
        if one:
          shape.append(ONE)
        elif fixed[k] is not None:
          shape.append(Const(fixed[k], I))
        else:
          self.not_one.add(v.uid)
          self.param_origin[v.uid] = (source, (leaf, k))
          shape.append(v)
      if layout == "C":
        stride_vars = None
        with self.in_entry():
          strides = self.c_strides(shape)
      else:
        stride_vars = tuple(Var(f"{name}_s{k}", I) for k in range(rank))
        strides = stride_vars
      param = ArrayParam(buf, off, shape_vars, stride_vars)
      self.params.append(param)
      self.array_params.append(param)
      root = Root(None, name, param)
      arr = ArrayVal(
                elt, tuple(shape), buf=buf, offset=off, strides=strides,
                roots=frozenset({root}), contig=layout == "C",
            )  # fmt: skip
      arr.inj = layout == "C"
      return arr
    if tag == "rec":
      return key[1].build(self.value_from_key(key[2], source, pos, name))
    if tag == "t":
      return tuple(self.value_from_key(k, source, pos, f"{name}_{i}") for i, k in enumerate(key[1:]))
    if tag == "n":
      return None
    if tag == "str":
      return key[1]
    if tag == "e":
      return Ellipsis
    if tag == "sl":
      parts = [self.value_from_key(k, source, pos, name) for k in key[1:]]
      return slice(*parts)
    if tag == "r":
      parts = [self.value_from_key(("s", WeakInt), source, pos, name) for _ in range(3)]
      return RangeVal(*parts)
    if tag == "ty":
      return self.type_val(key[1])
    if tag == "fn":
      code, cells = key[1], key[3]
      if isinstance(cells, Ref):
        return PyFunc(cells.obj, {})
      fn = CODE_FUNCTIONS[code]
      cell_values = {}
      for cname, ckey in cells:
        if ckey == ("unbound",):
          cell_values[cname] = Undefined(cname, "is a closure variable that has not been assigned yet")
        else:
          cell_values[cname] = self.value_from_key(ckey, source, pos, cname)
      return PyFunc(fn, cell_values)
    if tag == "o":
      return key[1].obj
    raise InternalError(f"unknown key {key!r}")

  @contextlib.contextmanager
  def in_entry(self):
    saved = self.block
    self.block = self.entry_block
    try:
      yield
    finally:
      self.block = saved

  @staticmethod
  def type_val(t) -> TypeVal:
    if t is bool:
      return TypeVal(WeakBool)
    if t is int:
      return TypeVal(WeakInt)
    if t is float:
      return TypeVal(WeakFloat)
    return TypeVal(t)

  def implicit_value(self, key: tuple, getter, obj):
    """The value of a global / closure variable / default object.

    Data (scalars, arrays) becomes a hidden runtime parameter that is
    re-read on every call; everything else is guarded by identity."""
    hit = self.implicit_cache.get(key, _MISSING)
    if hit is not _MISSING:
      return hit
    leaves: list = []
    try:
      k = classify(obj, leaves)
    except Unsupported:
      self.implicit_cache[key] = obj
      return obj
    static = frozenset(i for kk, i in self.state.static_implicit if kk == key)
    k2 = with_static_leaves(k, leaves, static)
    self.implicits.append(Implicit(key, getter, k2, static))
    value = self.value_from_key(k2, ("imp", key), [0], key[-1] if isinstance(key[-1], str) else "g")
    self.implicit_cache[key] = value
    return value

  def lift(self, obj):
    """Convert a Python object met at compile time into a value."""
    if isinstance(obj, bool):
      return Const(obj, WeakBool)
    if isinstance(obj, int):
      return Const(int(obj), WeakInt)
    if isinstance(obj, float):
      return Const(obj, WeakFloat)
    if obj is None or isinstance(obj, str) or obj is Ellipsis:
      return obj
    if isinstance(obj, np.generic):
      try:
        tv = type_value(obj.dtype)
      except Unsupported as e:
        raise self.error(UnsupportedError, str(e)) from None
      return Const(obj.item(), tv)
    if record := record_type(obj):
      return record.build(self.lift(x) for x in record.values(obj))
    if isinstance(obj, tuple):
      return tuple(self.lift(x) for x in obj)
    try:
      tv = type_value(obj)
    except Unsupported as e:
      raise self.error(UnsupportedError, str(e)) from None
    if tv is not None:
      return self.type_val(tv)
    if hasattr(obj, "__parakeet_fn__"):
      obj = obj.__parakeet_fn__
    if isinstance(obj, pytypes.FunctionType) and obj in STAGED:
      return obj
    if isinstance(obj, (np.ndarray, bytes, bytearray, memoryview, mmap.mmap, pytypes.FunctionType, list)):
      return self.implicit_value(("const", id(obj)), lambda o=obj: o, obj)
    return obj

  def require_static(self, v, what: str):
    """Return the Python value of ``v``, which must be known at compile time."""
    if isinstance(v, Const):
      return v.value
    if isinstance(v, Expr):
      # scalar arguments and array dimensions this value is computed from
      # become part of the specialization key; then it folds to a constant
      new_leaves: list = []
      new_implicit: list = []
      visited: set[int] = set()
      stack, seen, ok = [v], 0, True
      while stack:
        seen += 1
        if seen > 1000:
          ok = False
          break
        e = stack.pop()
        if isinstance(e, Var):
          if e.uid in visited:
            continue
          visited.add(e.uid)
          origin = self.param_origin.get(e.uid)
          if origin is not None:
            source, index = origin
            if source == "arg":
              new_leaves.append(index)
            else:
              new_implicit.append((source[1], index))
          elif e.uid in self.pure_defs:
            stack.append(self.pure_defs[e.uid])
          else:
            ok = False
            break
        elif isinstance(e, (ops.Prim, ops.Cast)):
          stack.extend(e.args if isinstance(e, ops.Prim) else (e.arg,))
        elif not isinstance(e, Const):
          ok = False
          break
      if ok and (new_leaves or new_implicit):
        self.state.static_leaves.update(new_leaves)
        self.state.static_implicit.update(new_implicit)
        self.restart("static argument")
    if isinstance(v, Expr):
      raise self.error(
        StaticValueError,
        f"{what} must be a compile-time constant",
        hint="use a literal, or a value computed only from literals and scalar arguments "
        "passed directly to the compiled function",
      )
    return v

  def static_int(self, v, what: str) -> int:
    if isinstance(v, ArrayVal) and v.rank == 0:
      v = self.elem(v, ())
    value = self.require_static(v, what)
    if isinstance(value, bool) or not isinstance(value, int):
      raise self.error(TypeInferenceError, f"{what} must be an integer, got {describe(v)}")
    return value

  def static_bool(self, v, what: str) -> bool:
    value = self.require_static(v, what)
    if not isinstance(value, (bool, int)):
      raise self.error(TypeInferenceError, f"{what} must be a boolean, got {describe(v)}")
    return bool(value)

  # ================================================================== entry

  def run(self) -> Elaborated:
    # variable names only need to be unique within one function; restarting
    # the numbering makes the generated source (and so the on-disk cache
    # key) independent of what was compiled before
    Var._counter = 0
    fn = self.root_fn
    src = get_source(fn)
    pos = [0]
    arg_values = [self.value_from_key(k, "arg", pos, n) for k, n in zip(self.arg_keys[1], self.arg_keys[0])]
    self.n_arg_leaves = pos[0]
    cells = {}
    for cname, ckey in self.arg_keys[2]:
      if ckey == ("unbound",):
        cells[cname] = Undefined(cname, "is a closure variable that has not been assigned yet")
      else:
        cells[cname] = self.value_from_key(ckey, "arg", pos, cname)
    root = PyFunc(fn, cells)
    try:
      result = self.call_user(root, [], {}, None, env=dict(zip(self.arg_keys[0], arg_values)))
      spec = self.export(result)
    except CompileError as e:
      if e.location is None and self.frames:
        e.location = self.loc()
      raise
    function = Function(
            _sanitize(fn.__name__), self.params, self.entry_block + self.body_block,
            self.results, self.fn_scope, src.location(),
        )  # fmt: skip
    return Elaborated(function, spec, self.implicits, self.writes_unknown)

  def export(self, v) -> tuple:
    """Turn the function's result into IR results plus a decoding spec."""
    if isinstance(v, Expr):
      self.results.append(self.tmp(v, "res"))
      return ("s", v.type)
    if isinstance(v, ArrayVal):
      c = self.as_concrete(v)
      self.emit(Incref(c.buf))
      self.results.append(ArrayResult(c.buf, c.offset, c.shape, c.strides, c.elt))
      return ("a", c.elt, c.rank)
    if record := record_type(v):
      return ("rec", record, self.export(record.values(v)))
    if isinstance(v, tuple):
      return ("t", *[self.export(x) for x in v])
    if v is None:
      return ("n",)
    if isinstance(v, str):
      return ("static", v)
    if isinstance(v, TypeVal):
      return ("static", v.type.dtype if not v.type.weak else v.type.python_type())
    if isinstance(v, RangeVal):
      for part in (v.start, v.stop, v.step):
        self.results.append(self.tmp(cast(part, WeakInt), "res"))
      return ("r",)
    if isinstance(v, Undefined):
      raise self.error(CompileError, f"result variable '{v.name}' {v.reason}")
    raise self.error(
      UnsupportedError,
      f"cannot return a {describe(v)} from compiled code to Python",
      hint="compiled functions can return scalars, arrays, tuples of those, or None",
    )

  # ================================================================ calling

  def call_user(self, fnval, args: list, kwargs: dict, node: ast.AST | None, env: dict | None = None):
    if isinstance(fnval, PyFunc):
      try:
        src = get_source(fnval.fn)
      except CompileError as e:
        if e.location is None:
          e.location, e.call_chain = self.loc(node), self.call_chain()
        raise
      fnode = src.node
      lexical = None
      cells = fnval.cells
      if cells is None:
        cells = self._lift_cells(fnval.fn)
        fnval.cells = cells
      gl = fnval.fn.__globals__
      name = fnval.fn.__name__
    else:
      src = fnval.frame.src
      fnode = fnval.node
      lexical = fnval.frame
      cells = None
      gl = fnval.frame.globals
      name = fnval.name
    if any(f.node is fnode for f in self.frames):
      raise self.error(
        UnsupportedError,
        f"recursive call to {name}()",
        node,
        hint="Parakeet inlines every call, so recursion cannot be compiled; rewrite it as a loop",
      )
    if len(self.frames) >= MAX_INLINE_DEPTH:
      raise self.error(UnsupportedError, f"calls nested more than {MAX_INLINE_DEPTH} deep", node)
    call_loc = self.loc(node) if self.frames else None
    if env is None:
      env = self.bind_arguments(fnval, fnode, name, args, kwargs, node)
    else:
      env = {k: self.bindable(v, k) for k, v in env.items()}
    sig = tuple(self.vtype(v) for v in env.values())
    parent_path = self.frame.path if self.frames else ()
    path = parent_path + ((id(node) if node is not None else 0, id(fnode), sig),)
    local_names = _local_names(fnode)
    sub_src = src if src.node is fnode else FunctionSource(fnode, src.filename, name)
    frame = Frame(
            sub_src, fnode, env, lexical, cells, gl, path, self.scope, self.block,
            len(self.block), call_loc, local_names,
        )  # fmt: skip
    facts = self.save_facts()
    self.frames.append(frame)
    depth = len(self.node_stack)
    try:
      if isinstance(fnode, ast.Lambda):
        result = self.eval(fnode.body)
      else:
        self.exec_block(fnode.body)
        ret = frame.ret
        if ret.target is not _MISSING:
          result = ret.target
        elif ret.done:
          result = ret.value
        else:
          result = None
    finally:
      del self.node_stack[depth:]
      self.frames.pop()
      if self.block is not frame.entry_block:
        self.block = frame.entry_block
        self.restore_facts(facts)
    return result

  def _lift_cells(self, fn) -> dict:
    cells = {}
    code = fn.__code__
    for cname, cell in zip(code.co_freevars, fn.__closure__ or ()):
      try:
        contents = cell.cell_contents
      except ValueError:
        cells[cname] = Undefined(cname, "is a closure variable that has not been assigned yet")
        continue
      if isinstance(
        contents,
        (
          bool,
          int,
          float,
          np.generic,
          np.ndarray,
          bytes,
          bytearray,
          memoryview,
          mmap.mmap,
          tuple,
          list,
          pytypes.FunctionType,
        ),
      ):
        cells[cname] = self.implicit_value(("cell", id(cell), cname), lambda c=cell: c.cell_contents, contents)
      else:
        cells[cname] = self.lift(contents)
    return cells

  def bind_arguments(self, fnval, fnode, name: str, args: list, kwargs: dict, node) -> dict:
    a: ast.arguments = fnode.args
    positional = [p.arg for p in a.posonlyargs + a.args]
    n_posonly = len(a.posonlyargs)
    env: dict[str, Any] = {}
    if len(args) > len(positional) and a.vararg is None:
      raise self.error(
        TypeInferenceError,
        f"{name}() takes {len(positional)} positional argument{'s' * (len(positional) != 1)} "
        f"but {len(args)} were given",
        node,
      )
    for pname, v in zip(positional, args):
      env[pname] = v
    if a.vararg is not None:
      env[a.vararg.arg] = tuple(args[len(positional) :])
    kwonly = [p.arg for p in a.kwonlyargs]
    for k, v in kwargs.items():
      if k in env and k in positional:
        raise self.error(TypeInferenceError, f"{name}() got multiple values for argument '{k}'", node)
      if (k in positional and positional.index(k) >= n_posonly) or k in kwonly:
        env[k] = v
      else:
        raise self.error(TypeInferenceError, f"{name}() got an unexpected keyword argument '{k}'", node)
    if a.kwarg is not None:
      raise self.error(UnsupportedError, f"{name}() takes **{a.kwarg.arg}, which is not supported", node)
    # defaults
    if isinstance(fnval, PyFunc):
      fn = fnval.fn
      n_def = len(fn.__defaults__ or ())
      first = len(positional) - n_def
      for i, pname in enumerate(positional):
        if pname not in env and i >= first:
          env[pname] = self._default(fn, i - first, None)
      for pname in kwonly:
        if pname not in env and fn.__kwdefaults__ and pname in fn.__kwdefaults__:
          env[pname] = self._default(fn, None, pname)
    else:
      first = len(positional) - len(fnval.defaults)
      for i, pname in enumerate(positional):
        if pname not in env and i >= first:
          env[pname] = fnval.defaults[i - first]
      for pname in kwonly:
        if pname not in env and pname in fnval.kw_defaults:
          env[pname] = fnval.kw_defaults[pname]
    missing = [p for p in positional + kwonly if p not in env]
    if missing:
      raise self.error(
        TypeInferenceError,
        f"{name}() missing {len(missing)} required argument{'s' * (len(missing) != 1)}: "
        + ", ".join(repr(m) for m in missing),
        node,
      )
    # parameters are variables: bind them like assignments
    order = positional + ([a.vararg.arg] if a.vararg else []) + kwonly
    return {p: self.bindable(env[p], p) for p in order}

  def _default(self, fn, index, kwname):
    if index is not None:
      obj = fn.__defaults__[index]
      key = ("default", id(fn.__code__), index)

      def getter(fn=fn, index=index):
        return fn.__defaults__[index]

    else:
      obj = fn.__kwdefaults__[kwname]
      key = ("kwdefault", id(fn.__code__), kwname)

      def getter(fn=fn, kwname=kwname):
        return fn.__kwdefaults__[kwname]

    if isinstance(obj, (np.ndarray, bytes, bytearray, memoryview, mmap.mmap, pytypes.FunctionType, list)) or hasattr(
      obj, "__parakeet_fn__"
    ):
      return self.implicit_value(key, getter, obj)
    return self.lift(obj)

  def call(self, fn, args: list, kwargs: dict | None = None, node: ast.AST | None = None):
    """Call any callable value."""
    kwargs = kwargs or {}
    if isinstance(fn, (PyFunc, Closure)):
      return self.call_user(fn, args, kwargs, node)
    from .. import lib

    if fn is RaggedArray:
      if len(args) + len(kwargs) != 2:
        raise self.error(TypeInferenceError, "RaggedArray() expects (values, offsets)", node)
      vals = args[0] if len(args) >= 1 else kwargs.get("values")
      offs = args[1] if len(args) >= 2 else kwargs.get("offsets")
      if vals is None or offs is None:
        raise self.error(TypeInferenceError, "RaggedArray() expects (values, offsets)", node)
      return RaggedArray(vals, self.as_array(offs, "RaggedArray offsets"))
    if isinstance(fn, type) and issubclass(fn, tuple) and hasattr(fn, "_fields"):
      try:
        result = fn(*args, **kwargs)
      except TypeError as e:
        raise self.error(TypeInferenceError, str(e), node) from None
      # Explicit arguments are already staged. Only Python-supplied defaults need lifting.
      return tuple.__new__(
        fn,
        (
          self.lift(value) if i >= len(args) and name not in kwargs else value
          for i, (name, value) in enumerate(zip(fn._fields, result))
        ),
      )
    if (
      isinstance(fn, type)
      and is_dataclass(fn)
      and getattr(fn, "__dataclass_params__", None) is not None
      and fn.__dataclass_params__.frozen
      and not fn.__module__.startswith("parakeet.")
    ):
      return self._construct_dataclass(fn, args, kwargs, node)
    if isinstance(fn, type) and issubclass(fn, enum.IntEnum):
      if len(args) != 1 or kwargs:
        raise self.error(TypeInferenceError, f"{fn.__name__}() takes 1 positional argument", node)
      val = self.as_scalar(args[0], f"argument of {fn.__name__}()")
      if val.type.kind not in "iub":
        raise self.error(TypeInferenceError, f"{fn.__name__}() expects an integer, got {describe(val)}", node)
      return cast(val, WeakInt)
    if isinstance(fn, BoundMethod):
      impl = lib.lookup_method(fn.obj, fn.name)
      if impl is None:
        raise self.error(UnsupportedError, f"method '{fn.name}' of {describe(fn.obj)} is not supported", node)
      return lib.invoke(self, impl, [fn.obj, *args], kwargs, fn.name)
    if isinstance(fn, TypeVal):
      return lib.invoke(self, lib.convert, [fn.type, *args], kwargs, fn.type.name)
    if isinstance(fn, Partial):
      return self.call(fn.fn, [*fn.args, *args], {**fn.kwargs, **kwargs}, node)
    if isinstance(fn, prims.UFunc):
      return lib.invoke(self, lib.call_ufunc, [fn, *args], kwargs, fn.name)
    impl = lib.lookup(fn)
    if impl is not None:
      return lib.invoke(self, impl, args, kwargs, getattr(fn, "__name__", "function"))
    if (isinstance(fn, pytypes.FunctionType) and fn not in STAGED) or hasattr(fn, "__parakeet_fn__"):
      return self.call(self.lift(fn), args, kwargs, node)
    staged = getattr(fn, "__parakeet_value__", None)
    if staged is not None:
      return self.call(staged(self), args, kwargs, node)
    if isinstance(fn, (Expr, ArrayVal, tuple)) or fn is None:
      raise self.error(TypeInferenceError, f"{describe(fn)} is not callable", node)
    raise self.error(UnsupportedError, lib.unsupported_message(fn), node)

  def _construct_dataclass(self, cls: type, args: list, kwargs: dict, node):
    if hasattr(cls, "__post_init__"):
      raise self.error(
        UnsupportedError,
        f"frozen dataclass {cls.__name__} with __post_init__ cannot be constructed inside compiled code",
        node,
      )
    dc_fields = fields(cls)
    pos_fields = [f for f in dc_fields if f.init and not f.kw_only]
    kw_fields = {f.name: f for f in dc_fields if f.init and f.kw_only}
    pos_names = {f.name: f for f in pos_fields}
    if len(args) > len(pos_fields):
      raise self.error(
        TypeInferenceError,
        f"{cls.__name__}() takes {len(pos_fields)} positional argument(s) but {len(args)} were given",
        node,
      )
    bound: dict[str, Any] = {}
    for f, v in zip(pos_fields, args):
      bound[f.name] = v
    for k, v in kwargs.items():
      if k in bound:
        raise self.error(TypeInferenceError, f"{cls.__name__}() got multiple values for argument '{k}'", node)
      if k in pos_names or k in kw_fields:
        bound[k] = v
      else:
        raise self.error(TypeInferenceError, f"{cls.__name__}() got an unexpected keyword argument '{k}'", node)
    values = []
    missing = []
    for f in dc_fields:
      if f.name in bound:
        values.append(bound[f.name])
      elif f.default is not MISSING:
        values.append(self.lift(f.default))
      elif f.default_factory is not MISSING:
        values.append(self.lift(f.default_factory()))
      elif f.init:
        missing.append(f.name)
      else:
        raise self.error(
          UnsupportedError, f"frozen dataclass {cls.__name__} field '{f.name}' has init=False and no default", node
        )
    if missing:
      raise self.error(
        TypeInferenceError,
        f"{cls.__name__}() missing {len(missing)} required argument(s): " + ", ".join(repr(m) for m in missing),
        node,
      )
    return Record(cls, tuple(f.name for f in dc_fields)).build(values)

  # =============================================================== statements

  def exec_block(self, stmts: list[ast.stmt]) -> bool:
    """Elaborate statements; True if control cannot fall off the end."""
    for stmt in stmts:
      self.node_stack.append(stmt)
      try:
        method = getattr(self, "exec_" + type(stmt).__name__, None)
        if method is None:
          raise self.error(
            UnsupportedError,
            f"'{_stmt_name(stmt)}' statements are not supported in compiled code",
            stmt,
          )
        try:
          terminated = method(stmt)
        except (Restart, ParakeetError):
          raise
        except RecursionError:
          raise self.error(CompileError, "expression nesting too deep to compile", stmt) from None
        except Exception as e:  # a bug in Parakeet: say where we were
          raise InternalError(
            f"{type(e).__name__}: {e}\nwhile compiling\n{self.loc(stmt).frame() if self.loc(stmt) else '?'}"
          ) from e
      except CompileError as e:
        if e.location is None:
          e.location, e.call_chain = self.loc(stmt), self.call_chain()
        raise
      finally:
        self.node_stack.pop()
      if terminated:
        return True
    return False

  def exec_Pass(self, node) -> bool:
    return False

  def exec_Expr(self, node: ast.Expr) -> bool:
    if isinstance(node.value, ast.Constant):
      return False  # docstring
    self.eval(node.value)
    return False

  def bindable(self, v, name: str):
    """Prepare a value for being held in a variable."""
    if isinstance(v, Expr) and not is_trivial(v):
      return self.tmp(v, _sanitize(name))
    if record := record_type(v):
      return record.build(self.bindable(x, name) for x in record.values(v))
    if isinstance(v, tuple):
      return tuple(self.bindable(x, name) for x in v)
    return v

  def bind(self, name: str, v) -> None:
    self.frame.env[name] = self.bindable(v, name)

  def assign(self, target: ast.expr, v) -> None:
    if isinstance(target, ast.Name):
      self.bind(target.id, v)
    elif isinstance(target, (ast.Tuple, ast.List)):
      if any(isinstance(e, ast.Starred) for e in target.elts):
        raise self.error(UnsupportedError, "starred assignment targets are not supported", target)
      parts = self.unpack(v, len(target.elts), target)
      for t, x in zip(target.elts, parts):
        self.assign(t, x)
    elif isinstance(target, ast.Subscript):
      obj = self.eval(target.value)
      index = self.eval_index(target.slice)
      if isinstance(obj, ArrayVal):
        if isinstance(v, tuple):
          v = self.as_array(v)
        self.setitem(obj, index, v)
      else:
        raise self.error(TypeInferenceError, f"{describe(obj)} does not support item assignment", target)
    elif isinstance(target, ast.Attribute):
      raise self.error(UnsupportedError, "assigning to attributes is not supported", target)
    else:
      raise self.error(UnsupportedError, "unsupported assignment target", target)

  def unpack(self, v, n: int, node=None) -> list:
    if isinstance(v, tuple):
      if len(v) != n:
        raise self.error(
          TypeInferenceError,
          f"cannot unpack {len(v)} value{'s' * (len(v) != 1)} into {n} target{'s' * (n != 1)}",
          node,
        )
      return list(v)
    if isinstance(v, ArrayVal) and v.rank >= 1:
      dim = v.shape[0]
      if isinstance(dim, Const):
        if dim.value != n:
          raise self.error(TypeInferenceError, f"cannot unpack array of length {dim.value} into {n} targets", node)
      else:
        self.check(cmp("eq", dim, n), "ValueError", f"cannot unpack array of length {{0}} into {n} targets", dim)
      return [self.getitem(v, Const(k, I)) for k in range(n)]
    raise self.error(TypeInferenceError, f"cannot unpack {describe(v)}", node)

  def exec_Assign(self, node: ast.Assign) -> bool:
    v = self.eval(node.value)
    if len(node.targets) > 1:
      v = self.bindable(v, "t")
    for target in node.targets:
      self.assign(target, v)
    return False

  def exec_AnnAssign(self, node: ast.AnnAssign) -> bool:
    if node.value is not None:
      self.assign(node.target, self.eval(node.value))
    return False

  def exec_AugAssign(self, node: ast.AugAssign) -> bool:
    op = _BINOPS.get(type(node.op))
    target = node.target
    if isinstance(target, ast.Name):
      cur = self.load_name(target.id, target)
      rhs = self.eval(node.value)
      if isinstance(cur, ArrayVal):
        # NumPy semantics: arrays are updated in place
        new = self.matmul(cur, rhs) if op is None else self.binop(op, cur, rhs)
        if not isinstance(new, ArrayVal) or new.rank != cur.rank:
          self.bind(target.id, new)
          return False
        if self.both_depth:
          raise self.no_derivative(
            "cannot differentiate an in-place update inside an if statement",
            node,
            hint=f"write `{target.id} = {target.id} {_SYMBOL.get(op, '@')} ...` so the branch makes a new array",
          )
        if not np.can_cast(new.elt.dtype, cur.elt.dtype, "same_kind"):
          raise self.error(
            TypeInferenceError,
            f"cannot cast the result of '{_SYMBOL.get(op, '@')}=' from {new.elt.name} to {cur.elt.name} in place",
            hint=f"write `{target.id} = {target.id} {_SYMBOL.get(op, '@')} ...` to create a new array",
          )
        self.force_inplace(cur)
        self.store_array(cur, new)
      else:
        self.bind(target.id, self.matmul(cur, rhs) if op is None else self.binop(op, cur, rhs))
    elif isinstance(target, ast.Subscript):
      obj = self.eval(target.value)
      if not isinstance(obj, ArrayVal):
        raise self.error(TypeInferenceError, f"{describe(obj)} does not support item assignment", target)
      index = self.bindable(self.eval_index(target.slice), "idx")
      self.force_inplace(obj)
      cur = self.getitem(obj, index)
      rhs = self.eval(node.value)
      new = self.matmul(cur, rhs) if op is None else self.binop(op, cur, rhs)
      self.setitem(obj, index, new)
    else:
      raise self.error(UnsupportedError, "unsupported augmented assignment target", target)
    return False

  def exec_FunctionDef(self, node: ast.FunctionDef) -> bool:
    for d in node.decorator_list:
      dv = self.eval(d)
      if not getattr(dv, "__parakeet_decorator__", False):
        raise self.error(UnsupportedError, "decorators on nested functions are not supported", d)
    self.frame.env[node.name] = self.make_closure(node, node.name)
    return False

  def make_closure(self, node, name: str) -> Closure:
    a = node.args
    defaults = [self.bindable(self.eval(d), "default") for d in a.defaults]
    kw_defaults = {
      p.arg: self.bindable(self.eval(d), "default") for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None
    }
    return Closure(node, self.frame, defaults, kw_defaults, name)

  def exec_Assert(self, node: ast.Assert) -> bool:
    cond = self.truth(self.eval(node.test), node.test)
    msg = ""
    if node.msg is not None:
      m = self.eval(node.msg)
      msg = m if isinstance(m, str) else "assertion failed"
    if isinstance(cond, Const) and cond.value:
      return False
    self.check(cond, "AssertionError", _escape(msg))
    return False

  def exec_Raise(self, node: ast.Raise) -> bool:
    exc_name, msg = "RuntimeError", ""
    if node.exc is not None:
      target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
      cls = self.eval(target)
      if not (isinstance(cls, type) and cls.__name__ in _EXCEPTIONS and _EXCEPTIONS[cls.__name__] is cls):
        raise self.error(
          UnsupportedError,
          "only built-in exceptions can be raised from compiled code: " + ", ".join(sorted(_EXCEPTIONS)),
          node,
        )
      exc_name = cls.__name__
      if isinstance(node.exc, ast.Call) and node.exc.args:
        m = self.eval(node.exc.args[0])
        if not isinstance(m, str):
          raise self.error(UnsupportedError, "exception messages must be string literals", node)
        msg = m
    self.check(Const(False, Bool), exc_name, _escape(msg))
    return True

  def exec_Delete(self, node: ast.Delete) -> bool:
    for t in node.targets:
      if not isinstance(t, ast.Name):
        raise self.error(UnsupportedError, "only plain variables can be deleted", t)
      self.frame.env[t.id] = Undefined(t.id, "was deleted")
    return False

  # ---- control flow ---------------------------------------------------------

  def exec_If(self, node: ast.If) -> bool:
    return self._conditional(node, self.truth(self.eval(node.test), node.test))

  def exec_Match(self, node: ast.Match) -> bool:
    subject = self.bindable(self.eval(node.subject), "match_subj")
    return self._exec_match_cases(node, subject, list(node.cases), 0)

  def _expand_match_or(self, pat: ast.pattern) -> list[ast.pattern]:
    """Expand nested MatchOr patterns that bind names into a flat list of alternatives."""
    if not self._pattern_binds_names(pat):
      return [pat]
    if isinstance(pat, ast.MatchOr):
      out: list[ast.pattern] = []
      for sub in pat.patterns:
        out.extend(self._expand_match_or(sub))
      return out
    if isinstance(pat, ast.MatchAs):
      if pat.pattern is None:
        return [pat]
      return [ast.MatchAs(pattern=sub, name=pat.name) for sub in self._expand_match_or(pat.pattern)]
    if isinstance(pat, ast.MatchSequence):
      choices = [self._expand_match_or(p) for p in pat.patterns]
      if all(len(c) == 1 for c in choices):
        return [pat]
      combos: list[list[ast.pattern]] = [[]]
      for c in choices:
        combos = [[*prev, item] for prev in combos for item in c]
      return [ast.MatchSequence(patterns=seq) for seq in combos]
    if isinstance(pat, ast.MatchClass):
      pos_choices = [self._expand_match_or(p) for p in pat.patterns]
      kw_choices = [self._expand_match_or(p) for p in pat.kwd_patterns]
      if all(len(c) == 1 for c in pos_choices) and all(len(c) == 1 for c in kw_choices):
        return [pat]
      pos_combos: list[list[ast.pattern]] = [[]]
      for c in pos_choices:
        pos_combos = [[*prev, item] for prev in pos_combos for item in c]
      kw_combos: list[list[ast.pattern]] = [[]]
      for c in kw_choices:
        kw_combos = [[*prev, item] for prev in kw_combos for item in c]
      return [
        ast.MatchClass(cls=pat.cls, patterns=ps, kwd_attrs=pat.kwd_attrs, kwd_patterns=ks)
        for ps in pos_combos
        for ks in kw_combos
      ]
    return [pat]

  def _exec_match_cases(self, node: ast.Match, subject, cases: list[ast.match_case], idx: int) -> bool:
    if idx >= len(cases):
      return False
    case = cases[idx]
    pat = case.pattern
    # Expand any top-level or nested OR patterns that bind names into sequential cases.
    if self._pattern_binds_names(pat):
      alts = self._expand_match_or(pat)
      if len(alts) > 1:
        expanded = [ast.match_case(pattern=sub, guard=case.guard, body=case.body) for sub in alts]
        return self._exec_match_cases(node, subject, cases[:idx] + expanded + cases[idx + 1 :], idx)

    fr = self.frame
    outer = self.block
    base_env = fr.env
    facts = self.save_facts()

    pat_cond, binds = self._match_pattern(pat, subject)
    if isinstance(pat_cond, Const) and not pat_cond.value:
      return self._exec_match_cases(node, subject, cases, idx + 1)

    if case.guard is None:
      if isinstance(pat_cond, Const) and pat_cond.value:
        for bname, bval in binds:
          self.bind(bname, bval)
        return self.exec_block(case.body)
      cond = self.tmp(pat_cond, "match_cond")
      then_blk: list = []
      else_blk: list = []
      fr.env = dict(base_env)
      self.block = then_blk
      for bname, bval in binds:
        self.bind(bname, bval)
      t_term = self.exec_block(case.body)
      then_env, then_end, then_facts = fr.env, self.block, self.save_facts()

      self.restore_facts(facts)
      fr.env = dict(base_env)
      self.block = else_blk
      e_term = self._exec_match_cases(node, subject, cases, idx + 1)
      else_env, else_end, else_facts = fr.env, self.block, self.save_facts()
      return self._finish_branch(
        outer,
        base_env,
        facts,
        cond,
        then_blk,
        then_env,
        then_end,
        then_facts,
        t_term,
        else_blk,
        else_env,
        else_end,
        else_facts,
        e_term,
      )

    # Guarded case: evaluate guard only when the pattern matched, with captures in scope.
    if isinstance(pat_cond, Const) and pat_cond.value:
      guard_env = dict(base_env)
      fr.env = guard_env
      for bname, bval in binds:
        self.bind(bname, bval)
      guard_cond = self.truth(self.eval(case.guard), case.guard)
      if isinstance(guard_cond, Const):
        if guard_cond.value:
          return self.exec_block(case.body)
        fr.env = base_env
        return self._exec_match_cases(node, subject, cases, idx + 1)
      cond = self.tmp(guard_cond, "guard_cond")
      then_blk = []
      else_blk = []
      fr.env = dict(guard_env)
      self.block = then_blk
      t_term = self.exec_block(case.body)
      then_env, then_end, then_facts = fr.env, self.block, self.save_facts()

      self.restore_facts(facts)
      fr.env = dict(base_env)
      self.block = else_blk
      e_term = self._exec_match_cases(node, subject, cases, idx + 1)
      else_env, else_end, else_facts = fr.env, self.block, self.save_facts()
      return self._finish_branch(
        outer,
        base_env,
        facts,
        cond,
        then_blk,
        then_env,
        then_end,
        then_facts,
        t_term,
        else_blk,
        else_env,
        else_end,
        else_facts,
        e_term,
      )

    # Runtime pattern condition AND guard: short-circuit guard inside If(pat_cond, ...).
    pat_cond = self.tmp(pat_cond, "match_cond")
    matched = Var("match_ok", Bool)
    self.mutables.add(matched.uid)
    self.emit(Assign(matched, ops.FALSE))
    guard_blk: list = []
    fr.env = dict(base_env)
    with self.in_block(guard_blk):
      for bname, bval in binds:
        self.bind(bname, bval)
      gcond = self.truth(self.eval(case.guard), case.guard)
      self.emit(Assign(matched, cast(gcond, Bool)))
    self.emit(If(pat_cond, guard_blk, []))

    then_blk = []
    else_blk = []
    fr.env = dict(base_env)
    self.block = then_blk
    for bname, bval in binds:
      self.bind(bname, bval)
    t_term = self.exec_block(case.body)
    then_env, then_end, then_facts = fr.env, self.block, self.save_facts()

    self.restore_facts(facts)
    fr.env = dict(base_env)
    self.block = else_blk
    e_term = self._exec_match_cases(node, subject, cases, idx + 1)
    else_env, else_end, else_facts = fr.env, self.block, self.save_facts()
    return self._finish_branch(
      outer,
      base_env,
      facts,
      matched,
      then_blk,
      then_env,
      then_end,
      then_facts,
      t_term,
      else_blk,
      else_env,
      else_end,
      else_facts,
      e_term,
    )

  def _finish_branch(
    self,
    outer: list,
    base_env: dict,
    facts,
    cond: Expr,
    then_blk: list,
    then_env: dict,
    then_end: list,
    then_facts,
    t_term: bool,
    else_blk: list,
    else_env: dict,
    else_end: list,
    else_facts,
    e_term: bool,
  ) -> bool:
    fr = self.frame
    outer.append(If(cond, then_blk, else_blk))
    if t_term and e_term:
      self.block, fr.env = outer, base_env
      self.restore_facts(facts)
      return True
    if t_term:
      self.block, fr.env = else_end, else_env
      self.restore_facts(else_facts)
      return False
    if e_term:
      self.block, fr.env = then_end, then_env
      self.restore_facts(then_facts)
      return False
    self.restore_facts(facts)
    fr.env = self.merge_envs(then_env, then_end, else_env, else_end, outer)
    self.block = outer
    return False

  @staticmethod
  def _pattern_binds_names(pat: ast.pattern) -> bool:
    for n in ast.walk(pat):
      if isinstance(n, ast.MatchAs) and n.name is not None:
        return True
      if isinstance(n, ast.MatchStar) and n.name is not None:
        return True
      if isinstance(n, ast.MatchMapping) and n.rest is not None:
        return True
    return False

  def _match_pattern(self, pat: ast.pattern, subject) -> tuple[Expr, list[tuple[str, Any]]]:
    if isinstance(pat, ast.MatchAs):
      if pat.pattern is None:
        cond: Expr = ops.TRUE
        binds: list[tuple[str, Any]] = []
      else:
        cond, binds = self._match_pattern(pat.pattern, subject)
      if pat.name is not None:
        binds = [*binds, (pat.name, subject)]
      return cond, binds
    if isinstance(pat, ast.MatchValue):
      val = self.eval(pat.value)
      if isinstance(subject, ArrayVal) or isinstance(val, ArrayVal):
        return ops.FALSE, []
      if isinstance(subject, (str, type(None))) != isinstance(val, (str, type(None))):
        return ops.FALSE, []
      if isinstance(subject, tuple) != isinstance(val, tuple):
        return ops.FALSE, []
      if (record_type(subject) is not None) != (record_type(val) is not None):
        return ops.FALSE, []
      return self.truth(self.compare(ast.Eq(), subject, val, pat), pat), []
    if isinstance(pat, ast.MatchSingleton):
      v = pat.value
      if v is None:
        return Const(subject is None, WeakBool), []
      if isinstance(subject, Expr) and subject.type.kind == "b":
        return (subject if v else ops.lnot(subject)), []
      return ops.FALSE, []
    if isinstance(pat, ast.MatchOr):
      acc: Expr = ops.FALSE
      for sub in pat.patterns:
        c, _ = self._match_pattern(sub, subject)
        if isinstance(c, Const):
          if c.value:
            return ops.TRUE, []
          continue
        acc = ops.lor(acc, c)
      return acc, []
    if isinstance(pat, ast.MatchSequence):
      return self._match_sequence(pat, subject)
    if isinstance(pat, ast.MatchClass):
      return self._match_class(pat, subject)
    raise self.error(UnsupportedError, f"pattern '{type(pat).__name__}' is not supported in match statements", pat)

  def _match_sequence(self, pat: ast.MatchSequence, subject) -> tuple[Expr, list[tuple[str, Any]]]:
    star_idx = next((i for i, p in enumerate(pat.patterns) if isinstance(p, ast.MatchStar)), None)
    if isinstance(subject, tuple):
      n = len(subject)
      if star_idx is None:
        if n != len(pat.patterns):
          return ops.FALSE, []
        elems = subject
        pats = pat.patterns
        binds: list[tuple[str, Any]] = []
      else:
        n_before = star_idx
        n_after = len(pat.patterns) - star_idx - 1
        if n < n_before + n_after:
          return ops.FALSE, []
        star_pat = pat.patterns[star_idx]
        assert isinstance(star_pat, ast.MatchStar)
        mid = subject[n_before : n - n_after if n_after else n]
        binds = [(star_pat.name, mid)] if star_pat.name is not None else []
        elems = subject[:n_before] + (subject[n - n_after :] if n_after else ())
        pats = pat.patterns[:n_before] + pat.patterns[star_idx + 1 :]
      cond: Expr = ops.TRUE
      for sub_p, elem in zip(pats, elems):
        c, b = self._match_pattern(sub_p, elem)
        if isinstance(c, Const) and not c.value:
          return ops.FALSE, []
        cond = ops.land(cond, c)
        binds.extend(b)
      return cond, binds
    if isinstance(subject, ArrayVal) and subject.rank >= 1:
      if star_idx is not None:
        raise self.error(UnsupportedError, "starred sequence patterns on arrays are not supported", pat)
      dim = subject.shape[0]
      if not isinstance(dim, Const):
        raise self.error(
          StaticValueError,
          "matching an array with a sequence pattern requires a compile-time length",
          pat,
        )
      if dim.value != len(pat.patterns):
        return ops.FALSE, []
      cond = ops.TRUE
      binds = []
      for k, sub_p in enumerate(pat.patterns):
        elem = self.getitem(subject, Const(k, I))
        c, b = self._match_pattern(sub_p, elem)
        if isinstance(c, Const) and not c.value:
          return ops.FALSE, []
        cond = ops.land(cond, c)
        binds.extend(b)
      return cond, binds
    return ops.FALSE, []

  def _match_class(self, pat: ast.MatchClass, subject) -> tuple[Expr, list[tuple[str, Any]]]:
    cls = self.eval(pat.cls)
    if isinstance(cls, TypeVal):
      t = cls.type
      if not t.weak:
        ok = isinstance(subject, Expr) and subject.type.kind == t.kind and subject.type.bits == t.bits
      else:
        ok = isinstance(subject, Expr) and (
          (t.kind == "b" and subject.type.kind == "b")
          or (t.kind in "iu" and subject.type.kind in "iu")
          or (t.kind == "f" and subject.type.kind == "f")
        )
      if not ok:
        return ops.FALSE, []
      if pat.kwd_attrs or len(pat.patterns) > 1:
        raise self.error(UnsupportedError, "scalar class patterns take at most one positional sub-pattern", pat)
      if pat.patterns:
        return self._match_pattern(pat.patterns[0], subject)
      return ops.TRUE, []
    rec = record_type(subject)
    if rec is None or not isinstance(cls, type) or not issubclass(rec.cls, cls):
      return ops.FALSE, []
    match_args = getattr(cls, "__match_args__", rec.names)
    if len(pat.patterns) > len(match_args):
      raise self.error(
        TypeInferenceError,
        f"{cls.__name__}() accepts {len(match_args)} positional sub-patterns ({len(pat.patterns)} given)",
        pat,
      )
    cond: Expr = ops.TRUE
    binds: list[tuple[str, Any]] = []
    for attr, sub_p in zip(match_args, pat.patterns):
      val = self.getattr(subject, attr, pat)
      c, b = self._match_pattern(sub_p, val)
      if isinstance(c, Const) and not c.value:
        return ops.FALSE, []
      cond = ops.land(cond, c)
      binds.extend(b)
    for attr, sub_p in zip(pat.kwd_attrs, pat.kwd_patterns):
      val = self.getattr(subject, attr, pat)
      c, b = self._match_pattern(sub_p, val)
      if isinstance(c, Const) and not c.value:
        return ops.FALSE, []
      cond = ops.land(cond, c)
      binds.extend(b)
    return cond, binds

  def _conditional(self, node: ast.If, cond: Expr) -> bool:
    if isinstance(cond, Const):
      return self.exec_block(node.body if cond.value else node.orelse)
    cond = self.tmp(cond, "cond")
    fr = self.frame
    if self.hint_get((fr.path, id(node), "both")):
      return self._if_both(node, cond)
    base_env = fr.env
    outer = self.block
    facts = self.save_facts()
    then_blk: list = []
    else_blk: list = []

    self.if_stack.append((fr, node))
    try:
      fr.env = dict(base_env)
      self.block = then_blk
      t_term = self.exec_block(node.body)
      then_env, then_end, then_facts = fr.env, self.block, self.save_facts()

      self.restore_facts(facts)
      fr.env = dict(base_env)
      self.block = else_blk
      e_term = self.exec_block(node.orelse)
      else_env, else_end, else_facts = fr.env, self.block, self.save_facts()
    finally:
      self.if_stack.pop()

    if t_term != e_term:
      # a guard: one branch never falls through.  If all it does is raise, what
      # follows always runs, so it continues in the enclosing block
      taken, taken_end = (else_blk, else_end) if t_term else (then_blk, then_end)
      if taken_end is taken and _only_raises(node.body if t_term else node.orelse):
        outer.append(If(cond, then_blk, []) if t_term else If(cond, [], else_blk))
        outer.extend(taken)
        self.block, fr.env = outer, else_env if t_term else then_env
        self.restore_facts(else_facts if t_term else then_facts)
        return False
    outer.append(If(cond, then_blk, else_blk))
    if t_term and e_term:
      self.block, fr.env = outer, base_env
      self.restore_facts(facts)
      return True
    if t_term:
      self.block, fr.env = else_end, else_env
      self.restore_facts(else_facts)
      return False
    if e_term:
      self.block, fr.env = then_end, then_env
      self.restore_facts(then_facts)
      return False
    self.restore_facts(facts)
    self.if_stack.append((fr, node))  # the merged variables are assigned on behalf of this statement
    try:
      fr.env = self.merge_envs(then_env, then_end, else_env, else_end, outer)
    finally:
      self.if_stack.pop()
    self.block = outer
    return False

  def request_both(self, fr: Frame, node: ast.If) -> None:
    """A branch computes differentiated values: start over and run both
    branches, choosing between their results (the derivative of a choice is
    the choice of the derivatives).  Returns if this ``if`` cannot be treated so."""
    key = (fr.path, id(node), "both")
    if not self.hint_get(key) and _selectable(node):
      self.state.hints[key] = True
      self.state.keep.append(node)
      self.restart("differentiated if statement")

  def _if_both(self, node: ast.If, cond: Expr) -> bool:
    fr = self.frame
    base = fr.env
    self.both_depth += 1
    try:
      fr.env = dict(base)
      self.exec_block(node.body)
      env_a = fr.env
      fr.env = dict(base)
      self.exec_block(node.orelse)
      env_b = fr.env
    finally:
      self.both_depth -= 1
    merged: dict[str, Any] = {}
    for name in env_a.keys() | env_b.keys():
      va, vb = env_a.get(name, _MISSING), env_b.get(name, _MISSING)
      if va is vb or (va is not _MISSING and vb is not _MISSING and self.static_equal(va, vb)):
        merged[name] = va
      elif va is _MISSING or vb is _MISSING or isinstance(va, Undefined) or isinstance(vb, Undefined):
        merged[name] = Undefined(name)
      else:
        merged[name] = self._choose(cond, va, vb, name)
    fr.env = merged
    return False

  def _choose(self, cond: Expr, a, b, name: str):
    if record := record_type(a):
      if record != record_type(b):
        return Undefined(name, "record types differ between branches")
      return record.build(self._choose(cond, record.values(a), record.values(b), name))
    if isinstance(a, tuple) and type(a) is type(b) and len(a) == len(b):
      return tuple(self._choose(cond, x, y, name) for x, y in zip(a, b))
    if isinstance(a, Expr) and isinstance(b, Expr):
      t = join(a.type, b.type)
      return self.tmp(ops.select(cond, cast(a, t), cast(b, t)), _sanitize(name))
    if isinstance(a, (Expr, ArrayVal)) and isinstance(b, (Expr, ArrayVal)):
      from ..lib.numpy_lib import where

      return where(self, cond, a, b)
    return Undefined(name, "holds different kinds of values on the two branches of a differentiated if statement")

  def merge_envs(self, env_a: dict, blk_a: list, env_b: dict, blk_b: list, outer: list) -> dict:
    merged: dict[str, Any] = {}
    for name in env_a.keys() | env_b.keys():
      va, vb = env_a.get(name, _MISSING), env_b.get(name, _MISSING)
      if va is vb:
        merged[name] = va
        continue
      if va is _MISSING or vb is _MISSING or isinstance(va, Undefined) or isinstance(vb, Undefined):
        merged[name] = Undefined(name)
        continue
      if self.static_equal(va, vb):
        merged[name] = va
        continue
      merged[name] = self.merge_values(name, va, blk_a, vb, blk_b)
    return merged

  def merge_values(self, name: str, va, blk_a: list, vb, blk_b: list):
    ta, tb = self.vtype(va), self.vtype(vb)
    vt = self.join_vtypes(ta, tb)
    if vt is None or vt[0] not in ("s", "a", "t", "sl", "rec", "n"):
      return Undefined(
        name,
        f"has incompatible types on different paths ({self.describe_vtype(ta)} vs {self.describe_vtype(tb)})",
      )
    if not self._mergeable(vt, va, vb):
      return Undefined(name, "holds different functions or objects on different paths")
    target = self.make_target(vt, va, name, self.scope)
    jvp = self.jvp
    if jvp is not None and not jvp.muted and (jvp.has_tangent(self, va) or jvp.has_tangent(self, vb)):
      jvp.make_tangent_target(self, target, name, self.scope)
    saved = self.block
    for blk, v in ((blk_a, va), (blk_b, vb)):
      self.block = blk
      self.assign_target(target, v)
    self.block = saved
    return target

  def _mergeable(self, vt, va, vb) -> bool:
    if vt[0] == "rec":
      return self._mergeable(vt[2], vt[1].values(va), vt[1].values(vb))
    if vt[0] in ("s", "a", "n"):
      return True
    if vt[0] == "t":
      return all(self._mergeable(t, x, y) for t, x, y in zip(vt[1:], va, vb))
    if vt[0] == "sl":
      return all(
        self._mergeable(t, x, y) for t, x, y in zip(vt[1:], (va.start, va.stop, va.step), (vb.start, vb.stop, vb.step))
      )
    return self.static_equal(va, vb)

  # ---- loops ----------------------------------------------------------------

  def begin_loop(self, node: ast.AST, names: frozenset[str]) -> LoopCtx:
    fr = self.frame
    ctx = LoopCtx(node, self.scope, self.block, self.new_scope("loop"))
    outside = fr.src.loads_outside(node) | {
      n.id for stmt in node.orelse for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    post = []
    for name in sorted(names):
      v = fr.env.get(name, _MISSING)
      if v is _MISSING or isinstance(v, Undefined):
        if name in outside:
          post.append(name)
        continue
      hint_key = (fr.path, id(node), name)
      vt = self.join_vtypes(self.vtype(v), self.hint_get(hint_key))
      if vt is None or vt[0] not in ("s", "a", "t", "sl", "rec"):
        ctx.static[name] = v  # functions, None, ...: must not change in the loop
        continue
      if self.carries_derivative(v):
        self.request_unroll(fr, node)
      target = self.make_target(vt, v, name, self.scope, hint_key)
      jvp = self.jvp
      if jvp is not None and not jvp.muted:
        if jvp.has_tangent(self, v) or self.hint_get((fr.path, id(node), name, "jvp")):
          jvp.make_tangent_target(self, target, name, self.scope)
      self.assign_target(target, v)
      ctx.carried[name] = (target, vt)
      fr.env[name] = target
    ctx.post_names = tuple(post)
    fr.loops.append(ctx)
    return ctx

  def sync(self, ctx: LoopCtx) -> None:
    """Write the current values of loop-carried variables to their homes."""
    fr = self.frame
    pending = []
    for name, (target, vt) in ctx.carried.items():
      v = fr.env.get(name, _MISSING)
      if v is target:
        continue
      if v is _MISSING or isinstance(v, Undefined):
        continue
      new = self.join_vtypes(vt, self.vtype(v))
      if new is None:
        raise self.error(
          TypeInferenceError,
          f"variable '{name}' changes type inside a loop: {self.describe_vtype(vt)} "
          f"becomes {self.describe_vtype(self.vtype(v))}",
        )
      if new != vt:
        self.state.hints[(fr.path, id(ctx.node), name)] = new
        self.restart("loop variable type")
      if self.carries_derivative(v):
        self.request_unroll(fr, ctx.node)
      jvp = self.jvp
      if jvp is not None and not jvp.muted and jvp.has_tangent(self, v):
        # If a carried variable acquired a tangent during the loop body, restart with a paired tangent target.
        def _needs_jvp_target(tgt) -> bool:
          if isinstance(tgt, Var):
            return tgt.type.kind == "f" and tgt.uid not in jvp.targets
          if isinstance(tgt, ArrayVal):
            return tgt.elt.kind == "f" and id(tgt) not in jvp.targets
          if rec := record_type(tgt):
            return _needs_jvp_target(rec.values(tgt))
          if isinstance(tgt, tuple):
            return any(_needs_jvp_target(x) for x in tgt)
          return False

        if _needs_jvp_target(target):
          self.state.hints[(fr.path, id(ctx.node), name, "jvp")] = True
          self.restart("jvp loop variable")
      pending.append((target, v))
    for name, before in ctx.static.items():
      v = fr.env.get(name, _MISSING)
      if v is not _MISSING and not self.static_equal(v, before):
        raise self.error(
          TypeInferenceError,
          f"variable '{name}' changes from {describe(before)} to {describe(v)} inside a loop",
          hint=f"initialise '{name}' before the loop with a value of its final type",
        )
    for name in ctx.post_names:
      v = fr.env.get(name, _MISSING)
      if v is _MISSING or isinstance(v, Undefined):
        continue
      vt_v = self.vtype(v)
      if vt_v[0] not in ("s", "a", "t", "sl", "rec"):
        continue
      hint_key = (fr.path, id(ctx.node), name)
      entry = ctx.post.get(name)
      if entry is None:
        vt = self.join_vtypes(vt_v, self.hint_get(hint_key))
        if vt is None:
          continue
        target = self.make_target(vt, v, name, ctx.home, hint_key, init_block=ctx.pre_block)
        entry = ctx.post[name] = (target, vt)
      target, vt = entry
      new = self.join_vtypes(vt, vt_v)
      if new is None:
        ctx.post[name] = (Undefined(name, "has different types on different loop iterations"), vt)
        continue
      if new != vt:
        self.state.hints[hint_key] = new
        self.restart("loop variable type")
      if not isinstance(target, Undefined):
        pending.append((target, v))
    if len(pending) > 1:
      pending = [(t, self.snapshot(v)) for t, v in pending]
    for target, v in pending:
      self.assign_target(target, v)

  def end_loop(self, ctx: LoopCtx) -> None:
    fr = self.frame
    fr.loops.pop()
    for name, (target, _) in ctx.carried.items():
      fr.env[name] = target
    for name in ctx.post_names:
      entry = ctx.post.get(name)
      if entry is None:
        fr.env.pop(name, None)
      else:
        fr.env[name] = entry[0]
    # names assigned in the body but neither carried nor needed afterwards
    for name in assigned_names(ctx.node.body, ctx.node) | _target_names(ctx.node):  # type: ignore[attr-defined]
      if name in ctx.static:
        fr.env[name] = ctx.static[name]
      elif name not in ctx.carried and name not in ctx.post_names and name in fr.env:
        fr.env[name] = Undefined(name, "is only assigned inside a loop that may not have run")
    if ctx.ret_seen:
      flag = fr.ret.flag
      if fr.loops:
        outer = fr.loops[-1]
        outer.ret_seen = True
        self.emit(If(flag, [ReleaseScope(outer.scope), Break()], []))
      else:
        rest: list = []
        self.emit(If(ops.lnot(flag), rest, []))
        self.block = rest

  def blame(self) -> str | None:
    """The library function whose own code (not a function passed to it) is
    being elaborated, for error messages."""
    best = None
    for name, depth in reversed(self.lib_calls):
      if name in ("gradient", "value_and_gradient"):
        break
      # every function entered since that call has to be library code
      if not all(f.globals.get("__parakeet_lib__") for f in self.frames[depth:]):
        break
      best = name
    return best

  def no_derivative(self, message: str, node: ast.AST | None = None, hint: str | None = None) -> CompileError:
    name = self.blame()
    if name is not None:
      return self.error(UnsupportedError, f"cannot differentiate through {name}(): its derivative is not implemented")
    return self.error(UnsupportedError, message, node, hint)

  def request_unroll(self, fr: Frame, node: ast.AST) -> None:
    """A loop turned out to update differentiated values: start over and
    unroll it, so that every iteration gets its own place on the tape."""
    key = (fr.path, id(node), "unroll")
    if not self.hint_get(key):
      self.state.hints[key] = True
      self.state.keep.append(node)
      self.restart("unroll differentiated loop")
    raise self.no_derivative(
      "cannot differentiate through this loop",
      node,
      hint="a loop that updates differentiated values has to be unrolled, which needs a 'for' loop over a "
      "range or tuple whose length is a compile-time constant; otherwise use parakeet.reduce, np.sum or "
      "another array operation",
    )

  def carries_derivative(self, v) -> bool:
    return self.tape is not None and not self.tape.muted and self.tape.is_active(self, v)

  def _unroll(self, node: ast.For, it) -> bool:
    from ..transforms import ad_loops

    what = "the bounds of a loop that updates differentiated values"
    # Small compile-time constant loops keep fast straight-line unrolling;
    # runtime-bound or large (> UNROLL_THRESHOLD) loops use IR checkpointed reverse loops.
    can_checkpoint = (
      self.tape is not None
      and not self.tape.muted
      and self.tape.outer is None
      and not _loop_transfers(node.body)
      and not node.orelse
    )
    if can_checkpoint:
      if (
        isinstance(it, RangeVal)
        and isinstance(it.start, Const)
        and isinstance(it.stop, Const)
        and isinstance(it.step, Const)
        and not isinstance(it.start.value, bool)
        and isinstance(it.start.value, int)
        and not isinstance(it.stop.value, bool)
        and isinstance(it.stop.value, int)
        and not isinstance(it.step.value, bool)
        and isinstance(it.step.value, int)
        and it.step.value != 0
      ):
        r = range(it.start.value, it.stop.value, it.step.value)
        if len(r) <= ad_loops.UNROLL_THRESHOLD:
          items: tuple = tuple(Const(i, WeakInt) for i in r)
          return self._unrolled_for(node, items)
      elif isinstance(it, RaggedArray) and isinstance(it.offsets.shape[0], Const):
        n_seg = it.offsets.shape[0].value - 1
        items = tuple(self._ragged_getitem(it, Const(k, I), node) for k in range(max(0, n_seg)))
        return self._unrolled_for(node, items)
      elif isinstance(it, ArrayVal) and it.rank >= 1 and isinstance(it.shape[0], Const):
        if it.shape[0].value <= ad_loops.UNROLL_THRESHOLD:
          items = tuple(self.getitem(it, Const(k, I)) for k in range(it.shape[0].value))
          return self._unrolled_for(node, items)
    else:
      try:
        if isinstance(it, RangeVal):
          r = range(*(self.static_int(x, what) for x in (it.start, it.stop, it.step)))
          if len(r) > _autodiff.MAX_UNROLL:
            raise self.no_derivative(
              f"cannot differentiate through a loop of {len(r)} iterations: it would have to be unrolled",
              node,
              hint="use an array operation, parakeet.map or parakeet.reduce instead",
            )
          items = tuple(Const(i, WeakInt) for i in r)
          return self._unrolled_for(node, items)
        if isinstance(it, RaggedArray) and isinstance(it.offsets.shape[0], Const):
          n_seg = it.offsets.shape[0].value - 1
          items = tuple(self._ragged_getitem(it, Const(k, I), node) for k in range(max(0, n_seg)))
          return self._unrolled_for(node, items)
        if isinstance(it, ArrayVal) and it.rank >= 1 and isinstance(it.shape[0], Const):
          items = tuple(self.getitem(it, Const(k, I)) for k in range(it.shape[0].value))
          return self._unrolled_for(node, items)
        raise self.error(StaticValueError, "not a range")
      except StaticValueError:
        raise self.no_derivative(
          "cannot differentiate through a loop whose length is only known at run time",
          node,
          hint="a loop that updates differentiated values has to be unrolled: loop over a range of a literal or "
          "of a scalar argument of the compiled function, or use parakeet.reduce, np.sum or another array operation",
        ) from None

    # IR-level O(sqrt(T)) checkpointed reverse for-loop
    end_checks: list = []
    if isinstance(it, RangeVal):
      start, stop, step = (self.tmp(x, n) for x, n in zip((it.start, it.stop, it.step), ("start", "stop", "step")))
      getter = None
    else:
      length, getter = self.iter_source(it, node.iter, end_checks)
      start, stop, step = ZERO, self.tmp(length, "n"), ONE
    names = assigned_names(node.body, node) | _target_names(node)
    return ad_loops.differentiate_for_loop(self, node, start, stop, step, getter, names, end_checks)

  def exec_While(self, node: ast.While) -> bool:
    if self.hint_get((self.frame.path, id(node), "unroll")):
      if (
        self.tape is not None
        and not self.tape.muted
        and self.tape.outer is None
        and not _loop_transfers(node.body)
        and not node.orelse
      ):
        from ..transforms import ad_loops

        names = assigned_names(node.body, node) | assigned_names(node.test)
        return ad_loops.differentiate_while_loop(self, node, names)
      self.request_unroll(self.frame, node)
    names = assigned_names(node.body, node) | assigned_names(node.test)
    ctx = self.begin_loop(node, names)
    self._track_exhaustion(ctx, node.orelse)
    blk: list = []
    with self.in_block(blk, ctx.scope, loop=True):
      cond = self.truth(self.eval(node.test), node.test)
      if not (isinstance(cond, Const) and cond.value):
        self.emit(If(ops.lnot(cond), [ReleaseScope(ctx.scope), Break()], []))
      if not self.exec_block(node.body):
        self.sync(ctx)
        self.emit(ReleaseScope(ctx.scope))
    self.emit(While(blk, ctx.scope))
    return self._finish_loop(ctx, node, [])

  def _name_used_after_loop(self, loop_node: ast.For, names: set[str]) -> bool:
    """Return True if any name in ``names`` may be read after ``loop_node`` without first
    being reassigned (conservative over parent blocks)."""

    def _walk_parent(parent: ast.AST) -> bool | None:
      for attr in ("body", "orelse", "finalbody"):
        stmts = getattr(parent, attr, None)
        if isinstance(stmts, list):
          for idx, st in enumerate(stmts):
            if st is loop_node:
              # Found loop_node in this statement list; check remaining statements in this block.
              # Conservatively refuse if any enclosing loop could re-read a name without reassigning.
              remaining = stmts[idx + 1 :]
              live = set(names)
              for rem in remaining:
                # If a subsequent `for x in ...` immediately rebinds `x` before reading it in its iter:
                if isinstance(rem, ast.For) and isinstance(rem.target, ast.Name) and rem.target.id in live:
                  iter_loads = {
                    n.id for n in ast.walk(rem.iter) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                  }
                  if iter_loads & live:
                    return True
                  live.discard(rem.target.id)
                  if not live:
                    return False
                  # Still check body for other names in `live`
                for sub in ast.walk(rem):
                  if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id in live:
                    return True
              return False
            res = _walk_parent(st)
            if res is not None:
              # If loop_node was nested inside another loop, fall back to loads_outside
              if isinstance(parent, (ast.For, ast.While)):
                return bool(names & self.frame.src.loads_outside(loop_node))
              return res
      return None

    res = _walk_parent(self.frame.src.node)
    if res is None:
      return bool(names & self.frame.src.loads_outside(loop_node))
    return res

  def _try_lift_scatter_loop(self, node: ast.For, it) -> bool:
    """Recognize ``for i in range(n): [b = expr; ...] out[idx_expr] += val_expr ...`` and lift
    directly to ``IndexScatter`` (privatized CPU / GPU atomic scatter-add, differentiable under
    grad/jvp, and automatically fused across sibling histograms sharing the same indices)."""
    if (
      node.orelse
      or not node.body
      or not isinstance(node.target, ast.Name)
      or not isinstance(it, RangeVal)
      or not (isinstance(it.start, Const) and it.start.value == 0)
      or not (isinstance(it.step, Const) and it.step.value == 1)
    ):
      return False
    loop_var = node.target.id
    bindings: list[tuple[str, ast.AST]] = []
    updates: list[tuple[str, ast.AST, ast.AST, list[tuple[str, ast.AST]]]] = []
    local_names: set[str] = {loop_var}
    out_names: list[str] = []
    out_name_set: set[str] = set()
    for stmt in node.body:
      if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        bname = stmt.targets[0].id
        if bname in local_names or bname in out_name_set:
          return False
        bindings.append((bname, stmt.value))
        local_names.add(bname)
      elif (
        isinstance(stmt, ast.AugAssign)
        and isinstance(stmt.op, ast.Add)
        and isinstance(stmt.target, ast.Subscript)
        and isinstance(stmt.target.value, ast.Name)
      ):
        out_name = stmt.target.value.id
        if out_name in local_names or out_name in out_name_set:
          return False
        out_names.append(out_name)
        out_name_set.add(out_name)
        updates.append((out_name, stmt.target.slice, stmt.value, list(bindings)))
      else:
        return False
    if not updates:
      return False
    if self._name_used_after_loop(node, local_names):
      return False
    # Ensure no output buffer name is read anywhere in bindings, indices, or values (no recurrence)
    exprs_to_check = [e for _, e in bindings]
    for _, idx_node, val_node, _ in updates:
      exprs_to_check.extend((idx_node, val_node))
    for expr_node in exprs_to_check:
      for sub in ast.walk(expr_node):
        if isinstance(sub, ast.Name) and sub.id in out_name_set:
          return False
    out_vals: dict[str, ArrayVal] = {}
    out_buf_uids: set[int] = set()
    for out_name in out_names:
      ov = self.frame.env.get(out_name, _MISSING)
      if not isinstance(ov, ArrayVal) or ov.rank != 1:
        return False
      if ov.buf is None:
        return False
      if ov.buf.uid in out_buf_uids or self.get_roots(ov) is None:
        return False
      out_buf_uids.add(ov.buf.uid)
      out_vals[out_name] = ov

    # Trial-probe bindings + index & value expressions to confirm scalar int index, scalar value, and no aliasing
    n = self.tmp(ops.imax(it.stop, ZERO), "n")
    probe_i = Var("i", I)
    self.nonneg.add(probe_i.uid)
    self.upper[probe_i.uid] = n
    saved_env = dict(self.frame.env)
    self.loop_depth += 1
    try:
      with self.trial() as tblk:
        self.frame.env[loop_var] = probe_i
        for bname, bexpr in bindings:
          bval = self.eval(bexpr)
          if not isinstance(bval, Expr):
            return False
          self.frame.env[bname] = bval
        probe_exprs: list[Expr] = []
        for _, idx_node, val_node, _ in updates:
          idx_p = self.eval(idx_node)
          val_p = self.eval(val_node)
          if not isinstance(idx_p, Expr) or idx_p.type.kind not in "iu":
            return False
          if not isinstance(val_p, Expr):
            return False
          probe_exprs.extend((idx_p, val_p))
        for st_ir in walk_stmts(tblk):
          if isinstance(st_ir, (Store, IndexScatter)):
            return False
        for expr_p in probe_exprs:
          for sub_e in walk_expr(expr_p):
            if isinstance(sub_e, Load) and sub_e.buf.uid in out_buf_uids:
              return False
    except Exception:  # noqa: BLE001
      return False
    finally:
      self.frame.env = saved_env
      self.loop_depth -= 1

    # Lower cleanly to IndexScatter (via ufunc_at, which handles grad + jvp and sibling scatter fusion!)
    ctx = self.capture()
    from ..lib.numpy_lib import ufunc_at

    def _same_dim(a: Expr, b: Expr) -> bool:
      return (isinstance(a, Var) and isinstance(b, Var) and a.uid == b.uid) or (
        isinstance(a, Const) and isinstance(b, Const) and a.value == b.value
      )

    def _direct_1d_slice(expr_n: ast.AST, bnds: list[tuple[str, ast.AST]]) -> ArrayVal | None:
      # Unwrap trivial local binding aliases (e.g. `b = bin_ids[i]` -> `bin_ids[i]`)
      bnd_map = dict(bnds)
      cur = expr_n
      seen_b: set[str] = set()
      while isinstance(cur, ast.Name) and cur.id in bnd_map and cur.id not in seen_b:
        seen_b.add(cur.id)
        cur = bnd_map[cur.id]
      if (
        isinstance(cur, ast.Subscript)
        and isinstance(cur.value, ast.Name)
        and isinstance(cur.slice, ast.Name)
        and cur.slice.id == loop_var
      ):
        arr = self.frame.env.get(cur.value.id, _MISSING)
        if isinstance(arr, ArrayVal) and arr.rank == 1:
          if _same_dim(arr.shape[0], it.stop) or _same_dim(arr.shape[0], n):
            return arr
          self.check(
            ops.cmp("le", n, arr.shape[0]),
            "IndexError",
            "index {0} is out of bounds for axis 0 with size {1}",
            n,
            arr.shape[0],
          )
          return self._slice_leading(arr, ZERO, n)
      return None

    # Pre-resolve all direct 1-D slices (and hoist their bounds checks) before emitting
    # any IndexScatter so sibling scatters stay adjacent for scatter fusion.
    prepared: list[tuple[str, ast.AST, ast.AST, list[tuple[str, ast.AST]], ArrayVal | None, ArrayVal | None]] = []
    for out_name, idx_node, val_node, up_bindings in updates:
      prepared.append(
        (
          out_name,
          idx_node,
          val_node,
          up_bindings,
          _direct_1d_slice(idx_node, up_bindings),
          _direct_1d_slice(val_node, up_bindings),
        )
      )

    for out_name, idx_node, val_node, up_bindings, direct_idx, direct_val in prepared:
      out_val = out_vals[out_name]

      def _make_evals(idx_n, val_n, bnds, elt):
        idx_refs = {sub.id for sub in ast.walk(idx_n) if isinstance(sub, ast.Name)}
        val_refs = {sub.id for sub in ast.walk(val_n) if isinstance(sub, ast.Name)}

        def _needed(refs):
          need = set(refs)
          for bn, be in reversed(bnds):
            if bn in need:
              need.update(sub.id for sub in ast.walk(be) if isinstance(sub, ast.Name))
          return [(bn, be) for bn, be in bnds if bn in need]

        idx_bnds = _needed(idx_refs)
        val_bnds = _needed(val_refs)

        def _eval_idx(coord):
          with self.restored(ctx):
            self.frame.env[loop_var] = coord[0]
            for bn, be in idx_bnds:
              self.frame.env[bn] = self.eval(be)
            return cast(self.eval(idx_n), I)

        def _eval_val(coord):
          with self.restored(ctx):
            self.frame.env[loop_var] = coord[0]
            for bn, be in val_bnds:
              self.frame.env[bn] = self.eval(be)
            return cast(self.eval(val_n), elt)

        return _eval_idx, _eval_val

      eval_idx, eval_val = _make_evals(idx_node, val_node, up_bindings, out_val.elt)
      idx_arr = direct_idx if direct_idx is not None else self.delayed(I, (n,), eval_idx, cost=2)
      val_arr = direct_val if direct_val is not None else self.delayed(out_val.elt, (n,), eval_val, cost=2)
      ufunc_at(self, "add", out_val, idx_arr, val_arr)
    return True

  def exec_For(self, node: ast.For) -> bool:
    it = self.eval(node.iter)
    tuple_domain = isinstance(it, tuple)
    if tuple_domain and (not it or not _loop_transfers(node.body)):
      return self._unrolled_for(node, it)
    if self._try_lift_scatter_loop(node, it):
      return False
    if self.hint_get((self.frame.path, id(node), "unroll")):
      return self._unroll(node, it)
    end_checks = []
    if tuple_domain:
      length, getter = self._tuple_loop_domain(it, node)
      start, stop, step = ZERO, length, ONE
    elif isinstance(it, RangeVal):
      start, stop, step = (self.tmp(x, n) for x, n in zip((it.start, it.stop, it.step), ("start", "stop", "step")))
      length = None
      getter = None
    else:
      length, getter = self.iter_source(it, node.iter, end_checks)
      start, stop, step = ZERO, self.tmp(length, "n"), ONE
    names = assigned_names(node.body, node) | _target_names(node)
    ctx = self.begin_loop(node, names)
    self._track_exhaustion(ctx, node.orelse or end_checks)
    var = Var("i", I)
    if self.is_nonneg(start) and isinstance(step, Const) and step.value > 0:
      self.nonneg.add(var.uid)
    if isinstance(step, Const) and step.value > 0:
      self.upper[var.uid] = stop
    blk: list = []
    with self.in_block(blk, ctx.scope, loop=True):
      self.assign(node.target, var if getter is None else getter(var))
      if not self.exec_block(node.body):
        self.sync(ctx)
        self.emit(ReleaseScope(ctx.scope))
    if not isinstance(step, Const):
      self.check(cmp("ne", step, 0), "ValueError", "range() arg 3 must not be zero")
    self.emit(ForRange(var, start, stop, step, blk, ctx.scope))
    return self._finish_loop(ctx, node, end_checks)

  def _track_exhaustion(self, ctx, needed):
    if needed and any(isinstance(t, ast.Break) for t in _loop_transfers(ctx.node.body)):
      ctx.exhausted = Var("exhausted", Bool)
      self.mutables.add(ctx.exhausted.uid)
      self.emit(Assign(ctx.exhausted, ops.TRUE))

  def _finish_loop(self, ctx, node, end_checks):
    self.end_loop(ctx)
    if ctx.exhausted is None:
      self.block.extend(end_checks)
      return self.exec_block(node.orelse)
    if end_checks:
      self.emit(If(ctx.exhausted, end_checks, []))
    if node.orelse:
      key = (id(node), "else guard")
      guard = self.hint_get(key)
      if guard is None:
        guard = ast.copy_location(ast.If(test=ast.Constant(True), body=node.orelse, orelse=[]), node)
        self.state.hints[key] = guard
      return self._conditional(guard, ctx.exhausted)
    return False

  def _tuple_loop_domain(self, items, node):
    """Select a fixed product of scalar columns without promoting unlike fields."""
    if all(isinstance(x, Expr) and x.type is items[0].type for x in items):
      return Const(len(items), I), lambda i: self.tuple_index(items, i, node)
    first = items[0]
    if isinstance(first, tuple) and all(type(x) is type(first) and len(x) == len(first) for x in items):
      columns = [self._tuple_loop_domain(col, node)[1] for col in zip(*items)]
      record = record_type(first)
      pack = record.build if record else tuple
      return Const(len(items), I), lambda i: pack(get(i) for get in columns)
    raise self.error(UnsupportedError, "break/continue in a tuple loop requires matching scalar or record types", node)

  def _unrolled_for(self, node: ast.For, items: tuple) -> bool:
    if items and _loop_transfers(node.body):
      raise self.error(UnsupportedError, "break/continue in an unrolled differentiated loop is unsupported", node)
    for item in items:
      self.assign(node.target, item)
      if self.exec_block(node.body):
        return True
    return self.exec_block(node.orelse)

  def iter_source(self, v, node=None, end_checks=None):
    """``(length, getter)`` for a value being iterated at run time."""
    if isinstance(v, ArrayVal):
      if v.rank == 0:
        raise self.error(TypeInferenceError, "iteration over a 0-d array", node)
      if v.buf is None and v.cost > 24:
        v = self.as_concrete(v)
      self.consume(v)
      return v.shape[0], lambda i: self.axis_view(v, 0, i, consumed=True)
    if isinstance(v, RangeVal):
      start, step = self.tmp(v.start), self.tmp(v.step)
      return self.range_length(v), lambda i: iadd(start, imul(i, step))
    if isinstance(v, ZipVal):
      child_checks = [[] for _ in v.items]
      parts = [self.iter_source(x, node, checks) for x, checks in zip(v.items, child_checks)]
      length = parts[0][0]
      for other, _ in parts[1:]:
        length = ops.imin(length, other)
      if end_checks is not None:
        # On the final next(), zip visits children left to right. Strict zip
        # also probes remaining children when its first child is exhausted.
        live = ops.TRUE
        probe = ops.FALSE
        for i, ((n, _), checks) in enumerate(zip(parts, child_checks)):
          ended = cmp("eq", n, length)
          if checks:
            end_checks.append(If(ops.land(ops.lor(live, probe), ended), checks, []))
          if i == 0:
            probe = ops.land(v.strict, ended)
          else:
            probe = ops.land(probe, ended)
          live = ops.land(live, ops.lnot(ended))
        with self.in_block(end_checks):
          for n, _ in parts[1:]:
            self.check(
              ops.lor(ops.lnot(v.strict), cmp("eq", parts[0][0], n)),
              "ValueError",
              "zip() arguments have different lengths",
            )
      return self.tmp(length, "n"), lambda i: tuple(g(i) for _, g in parts)
    if isinstance(v, EnumerateVal):
      length, getter = self.iter_source(v.item, node, end_checks)
      start = self.tmp(v.start)
      return length, lambda i: (iadd(start, i), getter(i))
    if isinstance(v, StructuredView):
      if v.rank == 0:
        raise self.error(TypeInferenceError, "iteration over a 0-d structured array", node)
      return v.shape[0], lambda i: self._struct_index(v, i, node)
    if isinstance(v, RaggedArray):
      if self.options.bounds_check:
        self.check(cmp("ge", v.offsets.shape[0], ONE), "ValueError", "RaggedArray offsets must have length >= 1")
      n_seg = self.tmp(isub(v.offsets.shape[0], ONE), "nseg")
      return n_seg, lambda i: self._ragged_getitem(v, i, node)
    if isinstance(v, tuple):
      arr = self.as_array(v)
      return self.iter_source(arr, node)
    raise self.error(TypeInferenceError, f"{describe(v)} is not iterable", node)

  def range_length(self, r: RangeVal) -> Expr:
    start, stop, step = r.start, r.stop, r.step
    if isinstance(step, Const):
      if step.value == 0:
        raise self.error(StaticValueError, "range() arg 3 must not be zero")
      if step.value > 0:
        span = isub(stop, start)
        n = span if step.value == 1 else ops.idiv(iadd(span, step.value - 1), step)
      else:
        n = ops.idiv(iadd(isub(start, stop), -step.value - 1), -step.value)
      length = self.tmp(ops.imax(n, 0), "n")
    else:
      step = self.tmp(step)
      self.check(cmp("ne", step, 0), "ValueError", "range() arg 3 must not be zero")
      pos = cmp("gt", step, 0)
      mag = ops.select(pos, step, isub(0, step))
      dist = ops.select(pos, isub(stop, start), isub(start, stop))
      length = self.tmp(ops.imax(ops.idiv(iadd(dist, isub(mag, 1)), mag), 0), "n")
    if isinstance(length, Var):
      self.nonneg.add(length.uid)
    return length

  def _innermost_loop(self, node) -> LoopCtx:
    fr = self.frame
    if not fr.loops:
      raise self.error(CompileError, f"'{_stmt_name(node)}' outside loop", node)
    return fr.loops[-1]

  def exec_Break(self, node) -> bool:
    ctx = self._innermost_loop(node)
    if ctx.exhausted is not None:
      self.emit(Assign(ctx.exhausted, ops.FALSE))
    self.sync(ctx)
    self.emit(ReleaseScope(ctx.scope))
    self.emit(Break())
    return True

  def exec_Continue(self, node) -> bool:
    ctx = self._innermost_loop(node)
    self.sync(ctx)
    self.emit(ReleaseScope(ctx.scope))
    self.emit(Continue())
    return True

  def exec_Return(self, node: ast.Return) -> bool:
    v = self.eval(node.value) if node.value is not None else None
    fr = self.frame
    ret = fr.ret
    if self.block is fr.entry_block and not fr.loops and ret.target is _MISSING:
      ret.value, ret.done = v, True
      return True
    v = self.bindable(v, "ret")
    vt = self.vtype(v)
    hint_key = (fr.path, "ret")
    if ret.target is _MISSING:
      joined = self.join_vtypes(vt, self.hint_get(hint_key))
      if joined is None:
        joined = vt
      ret.vtype = joined
      if joined[0] in ("s", "a", "t", "sl", "rec"):
        ret.target = self.make_target(joined, v, "ret", fr.scope, hint_key)
      else:
        ret.target = v
    else:
      joined = self.join_vtypes(ret.vtype, vt)
      if joined is None or (
        joined[0] not in ("s", "a", "t", "sl", "rec", "n") and not self.static_equal(ret.target, v)
      ):
        raise self.error(
          TypeInferenceError,
          f"function returns values of incompatible types: {self.describe_vtype(ret.vtype)} "
          f"and {self.describe_vtype(vt)}",
          node,
        )
      if joined != ret.vtype:
        self.state.hints[hint_key] = joined
        self.restart("return type")
    if ret.vtype[0] in ("s", "a", "t", "sl", "rec"):
      self.assign_target(ret.target, v)
    if fr.loops:
      if ret.flag is None:
        ret.flag = Var("returned", Bool)
        self.mutables.add(ret.flag.uid)
        fr.entry_block.insert(fr.entry_pos, Assign(ret.flag, Const(False, Bool)))
      self.emit(Assign(ret.flag, Const(True, Bool)))
      inner = fr.loops[-1]
      inner.ret_seen = True
      self.emit(ReleaseScope(inner.scope))
      self.emit(Break())
    return True

  # ============================================================== expressions

  def eval(self, node: ast.expr):
    self.node_stack.append(node)
    try:
      method = getattr(self, "eval_" + type(node).__name__, None)
      if method is None:
        raise self.error(
          UnsupportedError, f"'{type(node).__name__}' expressions are not supported in compiled code", node
        )
      v = method(node)
      if id(node) in self.state.eager:
        v = self.force_eager(v)
      return v
    except CompileError as e:
      if e.location is None:
        e.location, e.call_chain = self.loc(node), self.call_chain()
      raise
    finally:
      self.node_stack.pop()

  def force_eager(self, v):
    if isinstance(v, ArrayVal):
      return self.as_concrete(v)
    if record := record_type(v):
      return record.build(self.force_eager(x) for x in record.values(v))
    if isinstance(v, tuple):
      return tuple(self.force_eager(x) for x in v)
    return v

  def eval_Constant(self, node: ast.Constant):
    v = node.value
    if isinstance(v, bytes):
      return self.lift(v)
    if isinstance(v, (bool, int, float)) or v is None or isinstance(v, str) or v is Ellipsis:
      if isinstance(v, int) and not isinstance(v, bool) and not -(2**63) <= v < 2**64:
        raise self.error(StaticValueError, f"integer literal {v} does not fit in 64 bits", node)
      return self.lift(v) if not (isinstance(v, int) and v >= 2**63) else Const(v, ScalarType("uint64"))
    raise self.error(UnsupportedError, f"{type(v).__name__} literals are not supported", node)

  def load_name(self, name: str, node=None):
    fr: Frame | None = self.frame
    last = fr
    while fr is not None:
      v = fr.env.get(name, _MISSING)
      if v is not _MISSING:
        if isinstance(v, Undefined):
          raise self.error(CompileError, f"variable '{name}' {v.reason}", node)
        return v
      if name in fr.locals:
        raise self.error(CompileError, f"local variable '{name}' is used before it is assigned", node)
      if fr.cells is not None and name in fr.cells:
        v = fr.cells[name]
        if isinstance(v, Undefined):
          raise self.error(CompileError, f"variable '{name}' {v.reason}", node)
        return v
      last = fr
      fr = fr.lexical
    g = last.globals
    if name in g:
      obj = g[name]
      if isinstance(obj, pytypes.ModuleType):
        return obj
      if g.get("__parakeet_lib__"):
        return self.lift(obj)
      return self.implicit_value(("global", id(g), name), lambda g=g, name=name: g[name], obj)
    b = builtins.__dict__
    if name in b:
      return self.lift(b[name])
    raise self.error(CompileError, f"name '{name}' is not defined", node)

  def eval_Name(self, node: ast.Name):
    return self.load_name(node.id, node)

  def eval_NamedExpr(self, node: ast.NamedExpr):
    v = self.bindable(self.eval(node.value), node.target.id)
    self.frame.env[node.target.id] = v
    return v

  def eval_Tuple(self, node: ast.Tuple):
    out = []
    for e in node.elts:
      if isinstance(e, ast.Starred):
        v = self.eval(e.value)
        out.extend(self.unpack_star(v, e))
      else:
        out.append(self.eval(e))
    return tuple(out)

  def unpack_star(self, v, node) -> list:
    if isinstance(v, tuple):
      return list(v)
    if isinstance(v, ArrayVal) and v.rank >= 1 and isinstance(v.shape[0], Const):
      return [self.getitem(v, Const(k, I)) for k in range(v.shape[0].value)]
    raise self.error(
      TypeInferenceError, f"cannot unpack {describe(v)} with '*': its length is not known at compile time", node
    )

  def eval_List(self, node: ast.List):
    from .. import lib

    items = self.eval_Tuple(node)  # type: ignore[arg-type]
    return lib.array_literal(self, list(items))

  def eval_Slice(self, node: ast.Slice):
    return slice(
      None if node.lower is None else self.eval(node.lower),
      None if node.upper is None else self.eval(node.upper),
      None if node.step is None else self.eval(node.step),
    )

  def eval_index(self, node: ast.expr):
    if isinstance(node, ast.Tuple):
      return tuple(self.eval(e) for e in node.elts)
    return self.eval(node)

  def eval_Subscript(self, node: ast.Subscript):
    obj = self.eval(node.value)
    index = self.eval_index(node.slice)
    return self.subscript(obj, index, node)

  def subscript(self, obj, index, node=None):
    if isinstance(obj, ArrayVal):
      if isinstance(index, tuple):
        index = tuple(self._index_item(x) for x in index)
      else:
        index = self._index_item(index)
      return self.getitem(obj, index)
    if isinstance(obj, StructuredView):
      if isinstance(index, str):
        if index not in obj.names:
          raise self.error(StaticValueError, f"structured array has no field {index!r}", node)
        return obj.field(index)
      return self._struct_index(obj, index, node)
    if isinstance(obj, StructuredRow):
      if not isinstance(index, str) or index not in obj.names:
        raise self.error(StaticValueError, f"structured row has no field {index!r}", node)
      return obj.field(index)
    if isinstance(obj, RaggedArray):
      return self._ragged_getitem(obj, index, node)
    if isinstance(obj, dict):
      if not isinstance(index, str) or index not in obj:
        raise self.error(StaticValueError, "dictionary lookup requires a known string key", node)
      return obj[index]
    if isinstance(obj, tuple):
      return self.tuple_index(obj, index, node)
    if isinstance(obj, RangeVal):
      return self.getitem(self.range_array(obj), index)
    if isinstance(obj, Expr):
      if index == () or index is Ellipsis:
        return obj
      raise self.error(TypeInferenceError, f"a scalar ({describe(obj)}) cannot be indexed", node)
    raise self.error(TypeInferenceError, f"{describe(obj)} is not subscriptable", node)

  def _struct_index(self, sv: StructuredView, index, node=None):
    if not sv._fields:
      raise self.error(UnsupportedError, "indexing a zero-field structured array is not supported", node)
    if isinstance(index, tuple):
      items = tuple(self._index_item(x) for x in index)
    else:
      items = (self._index_item(index),)
    # Expand Ellipsis against sv.rank (not each field's subarray rank!) so subarray
    # trailing axes are preserved untouched.
    n_ell = sum(1 for x in items if x is Ellipsis)
    if n_ell > 1:
      raise self.error(TypeInferenceError, "an index can only have a single ellipsis ('...')", node)
    if n_ell == 1:
      explicit = sum(1 for x in items if x is not Ellipsis and x is not None)
      fill = max(0, sv.rank - explicit)
      expanded = []
      for x in items:
        if x is Ellipsis:
          expanded.extend([slice(None)] * fill)
        else:
          expanded.append(x)
      items = tuple(expanded)
    idx_arg = items[0] if len(items) == 1 else items
    sliced = tuple(self.getitem(f, idx_arg) for f in sv._fields)
    sub_rank = sv._fields[0].rank - sv.rank
    first_rank = sliced[0].rank if isinstance(sliced[0], ArrayVal) else 0
    new_rank = first_rank - sub_rank
    if new_rank <= 0:
      return StructuredRow(sv.dtype, sv.names, sliced)
    return StructuredView(sv.dtype, sv.names, sliced, new_rank)

  def _slice_leading(self, v, start: Expr, stop: Expr):
    if isinstance(v, RaggedArray):
      sub_off = self.getitem(v.offsets, slice(start, iadd(stop, ONE), None))
      base = self.tmp(cast(self.elem(v.offsets, (start,)), I), "rbase")
      end = self.tmp(cast(self.elem(v.offsets, (stop,)), I), "rend")
      rebased = self.binop("sub", sub_off, cast(base, sub_off.elt))
      return RaggedArray(self._slice_leading(v.values, base, end), rebased)
    if isinstance(v, ArrayVal):
      return self.getitem(v, slice(start, stop, None))
    if rec := record_type(v):
      return rec.build(self._slice_leading(x, start, stop) for x in rec.values(v))
    raise self.error(TypeInferenceError, f"cannot slice {describe(v)} along leading axis")

  def _ragged_leading_len(self, v) -> Expr:
    if isinstance(v, RaggedArray):
      return isub(v.offsets.shape[0], ONE)
    if isinstance(v, ArrayVal):
      return v.shape[0]
    if rec := record_type(v):
      vals = rec.values(v)
      return self._ragged_leading_len(vals[0]) if vals else ZERO
    return ZERO

  def _ragged_getitem(self, r: RaggedArray, index, node=None):
    offsets = r.offsets
    if self.options.bounds_check:
      self.check(cmp("ge", offsets.shape[0], ONE), "ValueError", "RaggedArray offsets must have length >= 1")
    n_seg = self.tmp(isub(offsets.shape[0], ONE), "nseg")
    if isinstance(index, slice):
      if index.step is not None and not (isinstance(index.step, Const) and index.step.value == 1):
        raise self.error(UnsupportedError, "RaggedArray slicing only supports step == 1", node)
      s = ZERO if index.start is None else self.as_index(index.start, "slice start")
      e = n_seg if index.stop is None else self.as_index(index.stop, "slice stop")
      s = ops.imax(ZERO, ops.imin(ops.select(cmp("lt", s, ZERO), iadd(s, n_seg), s), n_seg))
      e = ops.imax(s, ops.imin(ops.select(cmp("lt", e, ZERO), iadd(e, n_seg), e), n_seg))
      return self._slice_leading(r, s, e)
    idx = self.normalize_index(self.as_index(index, "RaggedArray index"), n_seg)
    start = self.tmp(cast(self.elem(offsets, (idx,)), I), "rstart")
    stop = self.tmp(cast(self.elem(offsets, (iadd(idx, ONE),)), I), "rstop")
    if self.options.bounds_check:
      total = self._ragged_leading_len(r.values)
      self.check(cmp("ge", start, ZERO), "ValueError", "RaggedArray offset must be non-negative")
      self.check(cmp("le", start, stop), "ValueError", "RaggedArray offsets must be non-decreasing")
      self.check(cmp("le", stop, total), "ValueError", "RaggedArray offset exceeds len(values)")
    return self._slice_leading(r.values, start, stop)

  def _index_item(self, x):
    if isinstance(x, tuple):  # x[(1, 2), ...]: a sequence index is an integer array
      return self.as_array(x)
    if isinstance(x, RangeVal):
      return self.range_array(x)
    return x

  def tuple_index(self, tup: tuple, index, node=None):
    if isinstance(index, slice):
      parts = [
        None if p is None else self.static_int(p, "tuple slice bounds") for p in (index.start, index.stop, index.step)
      ]
      return tup[slice(*parts)]
    if isinstance(index, ArrayVal) and index.rank == 0:
      index = self.elem(index, ())
    if not isinstance(index, Expr) or index.type.kind not in "iub":
      raise self.error(TypeInferenceError, f"tuple indices must be integers, not {describe(index)}", node)
    if isinstance(index, Const):
      i = int(index.value)
      if not -len(tup) <= i < len(tup):
        raise self.error(StaticValueError, f"tuple index {i} out of range for tuple of length {len(tup)}", node)
      return tup[i]
    # run-time index: possible when all elements have one scalar type
    vts = [self.vtype(x) for x in tup]
    joined = vts[0] if vts else None
    for t in vts[1:]:
      joined = self.join_vtypes(joined, t) if joined is not None else None
    if joined is None or joined[0] != "s":
      return tup[self.static_int(index, "the index of a tuple with elements of different types")]
    i = self.normalize_index(index, Const(len(tup), I))
    out = cast(tup[-1], joined[1])
    for k in range(len(tup) - 2, -1, -1):
      out = ops.select(cmp("eq", i, k), cast(tup[k], joined[1]), out)
    return self.tmp(out, "elt")

  def eval_Attribute(self, node: ast.Attribute):
    obj = self.eval(node.value)
    return self.getattr(obj, node.attr, node)

  def getattr(self, obj, name: str, node=None):
    from .. import lib

    if isinstance(obj, (np.ufunc, prims.UFunc)) and name == "at":
      uf_name = obj.name if isinstance(obj, prims.UFunc) else obj.__name__
      from ..lib.adverbs import LibFunction
      from ..lib.numpy_lib import ufunc_at

      return LibFunction(f"{uf_name}.at", lambda el, a, indices, b=None, _op=uf_name: ufunc_at(el, _op, a, indices, b))
    if isinstance(obj, (ArrayVal, Expr)):
      prop = lib.lookup_property(name)
      if prop is not None:
        return prop(self, obj)
      if lib.lookup_method(obj, name) is not None:
        return BoundMethod(obj, name)
      raise self.error(
        UnsupportedError,
        f"attribute '{name}' of {describe(obj)} is not supported",
        node,
        hint=lib.suggest_attribute(name),
      )
    if isinstance(obj, StructuredView):
      if name == "shape":
        return obj.shape
      if name == "ndim":
        return Const(obj.rank, I)
      if name == "size":
        return ops.iprod(obj.shape)
      if name == "dtype":
        return obj.dtype
      if name in obj.names:
        return obj.field(name)
      raise self.error(TypeInferenceError, f"structured array has no attribute or field {name!r}", node)
    if isinstance(obj, StructuredRow):
      if name in obj.names:
        return obj.field(name)
      raise self.error(TypeInferenceError, f"structured row has no field {name!r}", node)
    if isinstance(obj, RaggedArray):
      if name == "values":
        return obj.values
      if name == "offsets":
        return obj.offsets
      if name == "num_segments":
        if self.options.bounds_check:
          self.check(cmp("ge", obj.offsets.shape[0], ONE), "ValueError", "RaggedArray offsets must have length >= 1")
        return isub(obj.offsets.shape[0], ONE)
      if name == "lengths":
        if self.options.bounds_check:
          self.check(cmp("ge", obj.offsets.shape[0], ONE), "ValueError", "RaggedArray offsets must have length >= 1")
        return self.binop(
          "sub",
          self.getitem(obj.offsets, slice(ONE, None, None)),
          self.getitem(obj.offsets, slice(None, Const(-1, I), None)),
        )
      raise self.error(TypeInferenceError, f"RaggedArray has no attribute {name!r}", node)
    if (record := record_type(obj)) and record.cls is not dict and name in record.names:
      return getattr(obj, name)
    if record is not None and issubclass(record.cls, tuple) and name == "_replace":
      return BoundMethod(obj, "_replace")
    if record is not None and record.cls.__module__ == "parakeet.index":
      if (record.cls.__name__ == "SortedIndex" and name in ("get", "range")) or (
        record.cls.__name__ == "FrozenHashTable" and name == "get"
      ):
        return BoundMethod(obj, name)
    if isinstance(obj, dict) or record is not None:
      raise self.error(TypeInferenceError, f"unsupported record attribute {name!r}", node)
    if isinstance(obj, (tuple, PyFunc, Closure, BoundMethod, RangeVal, TypeVal)) or obj is None:
      if isinstance(obj, TypeVal):
        if name == "itemsize":
          return Const(obj.type.itemsize, I)
        if name == "type":
          return obj
      raise self.error(TypeInferenceError, f"{describe(obj)} has no attribute '{name}'", node)
    try:
      value = getattr(obj, name)
    except AttributeError:
      raise self.error(CompileError, f"{describe(obj)} has no attribute '{name}'", node) from None
    return self.lift(value)

  # ---- operators ---------------------------------------------------------------

  def eval_BinOp(self, node: ast.BinOp):
    a = self.eval(node.left)
    b = self.eval(node.right)
    if isinstance(node.op, ast.MatMult):
      return self.matmul(a, b)
    return self.binop(_BINOPS[type(node.op)], a, b)

  def matmul(self, a, b):
    from .. import lib

    return lib.matmul(self, a, b)

  def binop(self, op: str, a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
      if op == "add" and isinstance(a, tuple) and isinstance(b, tuple):
        return a + b
      if op == "mul":
        tup, n = (a, b) if isinstance(a, tuple) else (b, a)
        if isinstance(n, Expr) and not isinstance(tup, Expr):
          return tup * self.static_int(n, "tuple repetition count")
      a = self.as_array(a) if isinstance(a, tuple) else a
      b = self.as_array(b) if isinstance(b, tuple) else b
    return self.apply_ufunc(prims.UFUNCS[op], [a, b], python=True, symbol=_SYMBOL[op])

  def operand(self, v, symbol: str = ""):
    """Scalar expression or array for use in an element-wise operation."""
    if isinstance(v, (Expr, ArrayVal)):
      return v
    if isinstance(v, RangeVal):
      return self.range_array(v)
    if isinstance(v, tuple):
      return self.as_array(v)
    what = f"operand of '{symbol}'" if symbol else "operand"
    raise self.error(TypeInferenceError, f"unsupported {what}: {describe(v)}")

  def apply_ufunc(self, uf: prims.UFunc, args: list, python: bool = False, symbol: str = ""):
    vals = [self.operand(a, symbol or uf.name) for a in args]
    if len(vals) != uf.nin:
      raise self.error(TypeInferenceError, f"{uf.name} takes {uf.nin} argument(s), got {len(vals)}")
    if not any(isinstance(v, ArrayVal) and v.rank > 0 for v in vals):
      # as in NumPy, arithmetic on 0-d arrays yields scalars
      vals = [self.tmp(self.elem(v, ()), "x") if isinstance(v, ArrayVal) else v for v in vals]
    types = tuple(v.elt if isinstance(v, ArrayVal) else v.type for v in vals)
    all_weak = all(t.weak for t in types)
    if python and all_weak and uf.kind == "pow" and types[0].kind != "f" and types[1].kind != "f":
      if isinstance(vals[1], Const) and vals[1].value < 0:
        types = (WeakFloat, WeakFloat)
    try:
      ins, out = prims.resolve(uf, types, python)
    except TypeError as e:
      shown = ", ".join(describe(v) for v in vals)
      raise self.error(
        TypeInferenceError, f"'{symbol or uf.name}' is not supported for operands ({shown}): {e}"
      ) from None
    if python and all_weak and uf.kind in ("div", "intdiv") and not any(isinstance(v, ArrayVal) for v in vals):
      vals[1] = self.tmp(vals[1])
      self.check(
                ops.prim("ne", (vals[1], Const(0, vals[1].type)), Bool), "ZeroDivisionError",
                "float division by zero" if out.kind == "f" and uf.kind == "div" else "division by zero",
            )  # fmt: skip
    if not any(isinstance(v, ArrayVal) for v in vals):
      return uf.emit(tuple(cast(v, t) for v, t in zip(vals, ins)), out)

    def f(*xs):
      return uf.emit(tuple(cast(x, t) for x, t in zip(xs, ins)), out)

    return self.elementwise(vals, f, out.strong, cost=uf.cost)

  def eval_UnaryOp(self, node: ast.UnaryOp):
    v = self.eval(node.operand)
    if isinstance(node.op, ast.Not):
      return ops.lnot(self.truth(v, node.operand))
    op = _UNARY[type(node.op)]
    if op == "pos" and isinstance(v, (Expr, ArrayVal)):
      return v
    if isinstance(v, Const) and op == "neg" and v.type.weak and v.type.kind != "b":
      return Const(-v.value, v.type)
    return self.apply_ufunc(prims.UFUNCS[op], [v], python=True, symbol=_SYMBOL[op])

  def truth(self, v, node=None) -> Expr:
    """Python truthiness as a boolean expression."""
    if isinstance(v, Expr):
      if v.type.kind == "b":
        return v
      return cast(v, WeakBool if v.type.weak else Bool)
    if isinstance(v, ArrayVal):
      if v.rank == 0:
        return cast(self.tmp(self.elem(v, ())), Bool)
      if all(isinstance(d, Const) for d in v.shape) and all(d.value == 1 for d in v.shape):
        return cast(self.tmp(self.elem(v, (ZERO,) * v.rank)), Bool)
      raise self.error(
        TypeInferenceError,
        "the truth value of an array with more than one element is ambiguous",
        node,
        hint="use np.any(...) or np.all(...) (or .any() / .all())",
      )
    if v is None:
      return ops.FALSE
    if isinstance(v, (tuple, str)):
      return Const(len(v) > 0, WeakBool)
    if isinstance(v, Undefined):
      raise self.error(CompileError, f"variable '{v.name}' {v.reason}", node)
    return ops.TRUE

  def eval_BoolOp(self, node: ast.BoolOp):
    is_and = isinstance(node.op, ast.And)
    result = self.eval(node.values[0])
    for sub in node.values[1:]:
      result = self._short_circuit(is_and, result, lambda: self.eval(sub), sub)
    return result

  def _short_circuit(self, is_and: bool, left, eval_right, right_node: ast.expr):
    cond = self.truth(left, right_node)
    if isinstance(cond, Const):
      take_right = bool(cond.value) == is_and
      return eval_right() if take_right else left
    cond = self.tmp(cond, "cond")
    blk: list = []
    skipped: list = []
    before = self.frame.env
    self.frame.env = dict(before)
    with self.in_block(blk):
      right = eval_right()
      end_blk = self.block
    self.frame.env = self.merge_envs(self.frame.env, end_blk, before, skipped, self.block)
    lt, rt = self.vtype(left), self.vtype(right)
    both_bool = lt[0] == "s" and rt[0] == "s" and lt[1].kind == "b" and rt[1].kind == "b"
    if not blk and not skipped and both_bool:
      out_t = WeakBool if (lt[1].weak and rt[1].weak) else Bool
      return ops.prim("and" if is_and else "or", (cond, right), out_t)
    vt = self.join_vtypes(lt, rt)
    if vt is None or vt[0] not in ("s", "a"):
      raise self.error(
        TypeInferenceError,
        f"operands of '{'and' if is_and else 'or'}' have incompatible types: {describe(left)} and {describe(right)}",
        right_node,
      )
    target = self.make_target(vt, left, "bool", self.scope)
    self.assign_target(target, left)
    saved = self.block
    self.block = end_blk
    self.assign_target(target, right)
    self.block = saved
    self.emit(If(cond, blk, skipped) if is_and else If(cond, skipped, blk))
    return target

  def eval_IfExp(self, node: ast.IfExp):
    cond = self.truth(self.eval(node.test), node.test)
    if isinstance(cond, Const):
      return self.eval(node.body if cond.value else node.orelse)
    cond = self.tmp(cond, "cond")
    a_blk: list = []
    b_blk: list = []
    base_env = self.frame.env
    self.frame.env = dict(base_env)
    with self.in_block(a_blk):
      a = self.eval(node.body)
      a_end = self.block
    a_env = self.frame.env
    self.frame.env = dict(base_env)
    with self.in_block(b_blk):
      b = self.eval(node.orelse)
      b_end = self.block
    self.frame.env = self.merge_envs(a_env, a_end, self.frame.env, b_end, self.block)
    ta, tb = self.vtype(a), self.vtype(b)
    if not a_blk and not b_blk and ta[0] == "s" and tb[0] == "s":
      t = join(ta[1], tb[1])
      return ops.select(cond, cast(a, t), cast(b, t))
    if self.static_equal(a, b):
      if a_blk or b_blk:
        self.emit(If(cond, a_blk, b_blk))
      return a
    merged = self.merge_values("ifexp", a, a_end, b, b_end)
    if isinstance(merged, Undefined):
      raise self.error(
        TypeInferenceError,
        f"both branches of a conditional expression must have compatible types, got {describe(a)} and {describe(b)}",
        node,
      )
    self.emit(If(cond, a_blk, b_blk))
    return merged

  def eval_Compare(self, node: ast.Compare):
    left = self.eval(node.left)

    def part(k, lhs):
      right_node = node.comparators[k]
      right = self.eval(right_node)
      if k + 1 == len(node.ops):
        return self.compare(node.ops[k], lhs, right, node)
      right = self.bindable(right, "cmp")
      condition = self.compare(node.ops[k], lhs, right, node)
      return self._short_circuit(True, condition, lambda: part(k + 1, right), right_node)

    return part(0, left)

  def logical_and(self, a, b):
    if isinstance(a, Expr) and isinstance(b, Expr):
      ta, tb = self.truth(a), self.truth(b)
      return ops.prim("and", (ta, tb), WeakBool if (ta.type.weak and tb.type.weak) else Bool)
    return self.apply_ufunc(prims.UFUNCS["logical_and"], [a, b])

  def compare(self, op: ast.cmpop, a, b, node=None):
    if isinstance(op, (ast.Is, ast.IsNot)):
      same = self._identical(a, b)
      return Const(same if isinstance(op, ast.Is) else not same, WeakBool)
    if isinstance(op, (ast.In, ast.NotIn)):
      from .. import lib

      res = cast(self.truth(lib.contains(self, b, a)), WeakBool)
      return res if isinstance(op, ast.In) else ops.lnot(res)
    name = _CMPOPS[type(op)]
    if (a is None or b is None or isinstance(a, (str, TypeVal)) or isinstance(b, (str, TypeVal))) and name in (
      "eq",
      "ne",
    ):
      eq = self.static_equal(a, b)
      return Const(eq if name == "eq" else not eq, WeakBool)
    if isinstance(a, tuple) and isinstance(b, tuple):
      if name in ("eq", "ne"):
        if len(a) != len(b):
          return Const(name == "ne", WeakBool)
        acc: Expr = ops.TRUE
        for x, y in zip(a, b):
          acc = ops.land(acc, self.truth(self.compare(ast.Eq(), x, y, node)))
        return acc if name == "eq" else ops.lnot(acc)
      raise self.error(UnsupportedError, "ordering comparisons between tuples are not supported", node)
    if (ra := record_type(a)) is not None and (rb := record_type(b)) is not None and name in ("eq", "ne"):
      if ra != rb:
        return Const(name == "ne", WeakBool)
      acc = ops.TRUE
      for x, y in zip(ra.values(a), rb.values(b)):
        acc = ops.land(acc, self.truth(self.compare(ast.Eq(), x, y, node)))
      return acc if name == "eq" else ops.lnot(acc)
    return self.apply_ufunc(prims.UFUNCS[name], [a, b], python=True, symbol=_SYMBOL[name])

  @staticmethod
  def _identical(a, b) -> bool:
    if a is b:
      return True
    if isinstance(a, TypeVal) and isinstance(b, TypeVal):
      return a.type is b.type
    if isinstance(a, Const) and isinstance(b, Const) and a.type.kind == "b" and b.type.kind == "b":
      return a.value == b.value
    return False

  # ---- calls, lambdas, comprehensions ---------------------------------------------

  def eval_Call(self, node: ast.Call):
    fn = self.eval(node.func)
    args: list = []
    for a in node.args:
      if isinstance(a, ast.Starred):
        args.extend(self.unpack_star(self.eval(a.value), a))
      else:
        args.append(self.eval(a))
    kwargs = {}
    for kw in node.keywords:
      if kw.arg is None:
        raise self.error(UnsupportedError, "'**' argument unpacking is not supported", node)
      kwargs[kw.arg] = self.eval(kw.value)
    return self.call(fn, args, kwargs, node)

  def eval_Lambda(self, node: ast.Lambda):
    return self.make_closure(node, "<lambda>")

  def _comprehension(self, node, elt: ast.expr):
    from .. import lib

    if len(node.generators) > 1:
      return self._cartesian_comprehension(node, elt)
    gen = node.generators[0]
    if gen.is_async:
      raise self.error(UnsupportedError, "async comprehensions are not supported", node)
    source = self.eval(gen.iter)
    fr = self.frame

    def bind_and(fn):
      def run(item):
        saved = dict(fr.env)
        try:
          self.assign(gen.target, item)
          return fn()
        finally:
          fr.env.clear()
          fr.env.update(saved)

      return run

    if isinstance(source, tuple):
      items = [
        bind_and(lambda: self.eval(elt))(x) for x in source if not gen.ifs or self._static_filter(gen, x, bind_and)
      ]
      return lib.array_literal(self, items)
    end_checks = []
    length, getter = self.iter_source(source, gen.iter, end_checks)
    body = bind_and(lambda: self.eval(elt))
    if gen.ifs:

      def keep():
        acc: Expr = ops.TRUE
        for cond in gen.ifs:
          acc = ops.land(acc, self.truth(self.eval(cond), cond))
        return acc

      flags = lib.build_map(self, (length,), lambda idx: bind_and(keep)(getter(idx[0])))
      picks = self.nonzero1d(flags)
      result = lib.build_map(self, picks.shape, lambda idx: body(getter(self.tmp(self.elem(picks, idx), "i"))))
    else:
      result = lib.build_map(self, (self.tmp(length, "n"),), lambda idx: body(getter(idx[0])))
    if end_checks:
      result = self.as_concrete(result)
      self.block.extend(end_checks)
    return result

  def _cartesian_comprehension(self, node, elt):
    """Stage independent iteration domains once and retain Python's flat order."""
    from .. import lib
    from ..syntax import ir

    bound = set()
    shape = []
    getters = []
    for gen in node.generators:
      if gen.ifs or gen.is_async:
        raise self.error(
          UnsupportedError, "Cartesian comprehensions require unfiltered synchronous iterators", gen.iter
        )
      if any(isinstance(n, ast.Name) and n.id in bound for n in ast.walk(gen.iter)):
        raise self.error(UnsupportedError, "Cartesian comprehensions require independent iteration domains", gen.iter)
      if any(isinstance(n, ast.NamedExpr) for n in ast.walk(gen.iter)):
        raise self.error(UnsupportedError, "assignments in Cartesian iteration domains are unsupported", gen.iter)
      setup = []
      checks = []
      with self.in_block(setup):
        length, getter = self.iter_source(self.eval(gen.iter), gen.iter, checks)
      # Inner iterators may never be entered. Only hoist scalar setup that
      # cannot raise, change visible state, or allocate temporary storage.
      if checks or (
        getters
        and any(
          not isinstance(s, Assign) or any(isinstance(e, (ir.RtCall, ir.Load)) for e in ir.walk_expr(s.value))
          for s in setup
        )
      ):
        raise self.error(UnsupportedError, "Cartesian iterator setup must be safe to hoist", gen.iter)
      self.block.extend(setup)
      shape.append(length)
      getters.append(getter)
      bound.update(n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name))
    if any(isinstance(n, ast.NamedExpr) for n in ast.walk(elt)):
      raise self.error(UnsupportedError, "assignments in Cartesian comprehension bodies are unsupported", elt)
    fr = self.frame

    def body(indices):
      saved = dict(fr.env)
      start = len(self.block)
      try:
        for gen, get, index in zip(node.generators, getters, indices):
          self.assign(gen.target, get(index))
        value = self.eval(elt)
        for s in ir.walk_stmts(self.block[start:]):
          if isinstance(s, (ir.Store, ir.IndexScatter)) or any(
            isinstance(e, ir.RtCall) and not e.pure for x in ir.stmt_exprs(s) for e in ir.walk_expr(x)
          ):
            raise self.error(UnsupportedError, "Cartesian comprehension bodies must be pure", elt)
        return value
      finally:
        fr.env.clear()
        fr.env.update(saved)

    mapped = lib.build_map(self, tuple(shape), body, "Cartesian comprehension")
    return self.reshape(mapped, (ops.iprod(shape), *mapped.shape[len(shape) :]), checked=True)

  def _static_filter(self, gen, item, bind_and) -> bool:
    def keep():
      for cond in gen.ifs:
        if not self.static_bool(self.truth(self.eval(cond), cond), "a filter over a tuple"):
          return False
      return True

    return bind_and(keep)(item)

  def eval_ListComp(self, node: ast.ListComp):
    return self._comprehension(node, node.elt)

  def eval_GeneratorExp(self, node: ast.GeneratorExp):
    return self._comprehension(node, node.elt)

  def eval_JoinedStr(self, node):
    raise self.error(UnsupportedError, "f-strings are not supported in compiled code", node)

  # ================================================================ coercions

  def as_scalar(self, v, what: str = "value") -> Expr:
    if isinstance(v, Expr):
      return v
    if type(v) in (bool, int, float):
      return self.lift(v)
    if isinstance(v, ArrayVal):
      if v.rank == 0:
        return self.tmp(self.elem(v, ()), "x")
      if all(isinstance(d, Const) and d.value == 1 for d in v.shape):
        return self.tmp(self.elem(v, (ZERO,) * v.rank), "x")
      raise self.error(TypeInferenceError, f"{what} must be a scalar, got {describe(v)}")
    raise self.error(TypeInferenceError, f"{what} must be a scalar, got {describe(v)}")

  def as_index(self, v, what: str = "index") -> Expr:
    v = self.as_scalar(v, what)
    if v.type.kind not in "iub":
      raise self.error(TypeInferenceError, f"{what} must be integers, got {describe(v)}")
    return cast(v, I)

  def as_array(self, v, what: str = "value") -> ArrayVal:
    from .. import lib

    if isinstance(v, ArrayVal):
      return v
    if isinstance(v, Expr):
      return self.scalar_to_array(v)
    if isinstance(v, tuple):
      return lib.array_literal(self, list(v))
    if isinstance(v, RangeVal):
      return self.range_array(v)
    raise self.error(TypeInferenceError, f"{what} must be an array, got {describe(v)}")

  def range_array(self, r: RangeVal) -> ArrayVal:
    start, step = self.tmp(r.start), self.tmp(r.step)
    n = self.range_length(r)
    from ..ndtypes import Int64

    return self.delayed(Int64, (n,), lambda idx: cast(iadd(start, imul(idx[0], step)), Int64), cost=1)

  def as_dtype(self, v, what: str = "dtype") -> ScalarType:
    """A strong element type from a dtype-like value."""
    if isinstance(v, TypeVal):
      return v.type.strong
    if isinstance(v, str):
      try:
        tv = type_value(np.dtype(v))
      except (TypeError, Unsupported) as e:
        raise self.error(TypeInferenceError, f"invalid {what} {v!r}: {e}") from None
      return tv
    if isinstance(v, (ArrayVal,)):
      return v.elt
    raise self.error(TypeInferenceError, f"{what} must be a dtype, got {describe(v)}")

  def shape_tuple(self, v, what: str = "shape") -> tuple[Expr, ...]:
    """Normalise an int / tuple / small array into a tuple of dimensions."""
    if isinstance(v, Expr):
      return (self.as_index(v, what),)
    if isinstance(v, tuple):
      return tuple(self.as_index(x, what) for x in v)
    if isinstance(v, ArrayVal):
      if v.rank == 0:
        return (self.as_index(self.elem(v, ()), what),)
      if v.rank == 1 and isinstance(v.shape[0], Const):
        if v.elt.kind not in "iu":
          raise self.error(TypeInferenceError, f"{what} must be integers, got {describe(v)}")
        return tuple(self.tmp(cast(self.elem(v, (Const(k, I),)), I), "dim") for k in range(v.shape[0].value))
      raise self.error(
        StaticValueError,
        f"the number of dimensions in {what} must be known at compile time",
        hint="pass a tuple such as (n, m) instead of an array",
      )
    raise self.error(TypeInferenceError, f"{what} must be an int or a tuple of ints, got {describe(v)}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_locals_cache: dict[int, tuple[ast.AST, frozenset[str]]] = {}


def _local_names(fnode) -> frozenset[str]:
  hit = _locals_cache.get(id(fnode))
  if hit is not None and hit[0] is fnode:
    return hit[1]
  if isinstance(fnode, ast.Lambda):
    names: frozenset[str] = frozenset()
  else:
    names = assigned_names(fnode.body)
  a = fnode.args
  params = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
  if a.vararg:
    params.add(a.vararg.arg)
  result = frozenset(names - params)
  _locals_cache[id(fnode)] = (fnode, result)
  return result


def _selectable(node: ast.If) -> bool:
  """Can both branches be run and their results chosen between?  Only if
  running the branch not taken has no effect: plain assignments to variables."""
  for stmt in node.body + node.orelse:
    for n in ast.walk(stmt):
      if isinstance(n, (ast.Return, ast.Break, ast.Continue, ast.Raise, ast.Assert, ast.While, ast.Delete)):
        return False
      if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant):
        return False
      if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        for t in targets:
          if any(isinstance(m, (ast.Subscript, ast.Attribute)) for m in ast.walk(t)):
            return False
  return True


def _only_raises(stmts: list[ast.stmt]) -> bool:
  """Does this branch end by raising, with no other way out?"""
  if not stmts or not isinstance(stmts[-1], ast.Raise):
    return False
  return not any(isinstance(n, (ast.Return, ast.Break, ast.Continue)) for s in stmts for n in ast.walk(s))


def _target_names(node) -> frozenset[str]:
  target = getattr(node, "target", None)
  if target is None:
    return frozenset()
  return frozenset(n.id for n in ast.walk(target) if isinstance(n, ast.Name))


def _loop_transfers(body):
  """Breaks and continues targeting this loop, including those in an inner loop's else."""
  pending = list(body)
  transfers = []
  while pending:
    stmt = pending.pop()
    if isinstance(stmt, (ast.Break, ast.Continue)):
      transfers.append(stmt)
    elif isinstance(stmt, (ast.For, ast.While)):
      pending.extend(stmt.orelse)
    elif not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
      pending.extend(ast.iter_child_nodes(stmt))
  return transfers


def _stmt_name(node) -> str:
  return {
        "Try": "try", "With": "with", "Global": "global", "Nonlocal": "nonlocal", "Import": "import",
        "ImportFrom": "import", "ClassDef": "class", "AsyncFunctionDef": "async def",
        "Match": "match", "Break": "break", "Continue": "continue", "TryStar": "try",
        "AsyncFor": "async for", "AsyncWith": "async with",
    }.get(type(node).__name__, type(node).__name__)  # fmt: skip


def _escape(msg: str) -> str:
  return msg.replace("{", "{{").replace("}", "}}")
