"""Registry of library functions plus the core building blocks they share.

A *staged implementation* is a Python function ``impl(el, *args, **kwargs)``
that receives compile-time values and uses the elaborator ``el`` to build IR.
Implementations are registered against the Python objects they stand for
(``np.zeros``, ``len``, ...) so user code keeps calling the ordinary names.
"""

from __future__ import annotations

import difflib
import inspect
import types as pytypes
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import ops, prims
from .. import signature as _signature
from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..ir import Assign, Const, Expr, RtCall, Store, Var, walk_expr, walk_stmts
from ..ir import Check as _Check
from ..ir import Comment as _Comment
from ..ndtypes import Bool, Float64, Int64, ScalarType, join
from ..ops import ZERO, I, cast, cmp
from ..values import CHEAP_COST, INF_COST, ArrayVal, PyFunc, RangeVal, TypeVal, describe

FUNCTIONS: dict[Any, Callable] = _signature.STAGED
METHODS: dict[str, Callable] = {}
PROPERTIES: dict[str, Callable] = {}
NAMES: dict[Any, str] = {}
_signatures: dict[Callable, inspect.Signature] = {}


def register(*targets, name: str | None = None):
  """Register a staged implementation for one or more Python callables."""

  def deco(impl):
    for t in targets:
      if t is None:
        continue
      FUNCTIONS[t] = impl
      NAMES[t] = name or getattr(t, "__name__", impl.__name__)
    return impl

  return deco


def method(*names):
  def deco(impl):
    for n in names:
      METHODS[n] = impl
    return impl

  return deco


def prop(*names):
  def deco(impl):
    for n in names:
      PROPERTIES[n] = impl
    return impl

  return deco


def source(*targets, name: str | None = None):
  """Register a plain-Python implementation that is compiled like user code."""

  def deco(fn):
    holder = PyFunc(fn, {})

    def impl(el, *args, **kwargs):
      return el.call_user(holder, list(args), kwargs, None)

    impl.__name__ = fn.__name__
    impl.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    impl.__parakeet_source__ = fn  # type: ignore[attr-defined]
    for t in targets:
      if t is None:
        continue
      FUNCTIONS[t] = impl
      NAMES[t] = name or getattr(t, "__name__", fn.__name__)
    fn.__parakeet_impl__ = impl
    return fn

  return deco


def lookup(obj) -> Callable | None:
  try:
    impl = FUNCTIONS.get(obj)
  except TypeError:  # unhashable
    return None
  if impl is not None:
    return impl
  if isinstance(obj, np.ufunc):
    uf = prims.BY_NUMPY.get(obj)
    if uf is not None:
      return lambda el, *args, **kwargs: call_ufunc(el, uf, *args, **kwargs)
  return None


def lookup_method(obj, name: str) -> Callable | None:
  return METHODS.get(name)


def lookup_property(name: str) -> Callable | None:
  return PROPERTIES.get(name)


def invoke(el, impl: Callable, args: list, kwargs: dict, name: str):
  """Call a staged implementation, turning argument mismatches into
  compile errors that name the user-visible function."""
  sig = _signatures.get(impl)
  if sig is None:
    try:
      sig = inspect.signature(impl)
    except (TypeError, ValueError):
      sig = None  # type: ignore[assignment]
    _signatures[impl] = sig  # type: ignore[assignment]
  if sig is not None:
    is_source = hasattr(impl, "__parakeet_source__")
    try:
      bound = sig.bind(*args, **kwargs) if is_source else sig.bind(el, *args, **kwargs)
    except TypeError as e:
      raise el.error(TypeInferenceError, f"{name}(): {e}") from None
    if not is_source:
      # default values are plain Python objects: turn numbers into
      # compile-time constants so implementations only ever see values
      given = bound.arguments
      extra = {}
      for pname, p in sig.parameters.items():
        if pname not in given and p.default is not p.empty and type(p.default) in (bool, int, float):
          if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            extra[pname] = el.lift(p.default)
      if extra:
        return impl(*bound.args, **{**bound.kwargs, **extra})
  return impl(el, *args, **kwargs)


def unsupported_message(fn) -> str:
  mod = getattr(fn, "__module__", None) or ""
  name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
  full = f"{mod}.{name}" if mod and mod != "builtins" else name
  if isinstance(fn, pytypes.ModuleType):
    return f"module '{fn.__name__}' is not callable"
  known = sorted({n for n in NAMES.values()})
  close = difflib.get_close_matches(name.split(".")[-1], known, n=3, cutoff=0.7)
  msg = f"{full} is not supported in compiled code"
  if close:
    msg += f" (supported functions with similar names: {', '.join(close)})"
  return msg


def suggest_attribute(name: str) -> str | None:
  close = difflib.get_close_matches(name, sorted(set(METHODS) | set(PROPERTIES)), n=3, cutoff=0.7)
  return f"did you mean: {', '.join(close)}?" if close else None


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


def convert(el, type: ScalarType, x=None):
  """``float(x)``, ``np.float32(x)``, ``int(x)``... for scalars and arrays."""
  if x is None:
    return Const(0, type)
  if isinstance(x, tuple):
    x = el.as_array(x)
  if isinstance(x, RangeVal):
    x = el.range_array(x)
  if isinstance(x, ArrayVal):
    if type.weak:  # float(arr) / int(arr): only for single-element arrays
      return to_python_scalar(el, type, el.as_scalar(x, f"argument of {type.python_type().__name__}()"))
    return el.cast_array(x, type)
  if isinstance(x, Expr):
    if type.weak:
      return to_python_scalar(el, type, x)
    return cast(x, type)
  raise el.error(TypeInferenceError, f"cannot convert {describe(x)} to {type.name}")


def to_python_scalar(el, type: ScalarType, x: Expr) -> Expr:
  if type.kind == "i" and x.type.kind == "f":
    # int() truncates and rejects non-finite values
    x = el.tmp(x)
    el.check(ops.prim("isfinite", (x,), Bool), "ValueError", "cannot convert float NaN or infinity to integer")
  return cast(x, type)


def call_ufunc(el, uf: prims.UFunc, *args, dtype=None, out=None, where=None):
  if out is not None or where is not None:
    raise el.error(UnsupportedError, f"the 'out' and 'where' arguments of {uf.name} are not supported")
  res = el.apply_ufunc(uf, list(args), python=False)
  if dtype is not None:
    res = convert(el, el.as_dtype(dtype), res)
  return res


# ---------------------------------------------------------------------------
# array literals
# ---------------------------------------------------------------------------


def array_literal(el, items: list, dtype: ScalarType | None = None) -> ArrayVal:
  """``[a, b, c]`` / ``np.array((a, b, c))``: stack scalars or arrays."""
  if not items:
    return el.alloc_array(dtype or Float64, (ZERO,))
  items = [el.as_array(x) if isinstance(x, (tuple, RangeVal)) else x for x in items]
  if all(isinstance(x, Expr) for x in items):
    elt = dtype
    if elt is None:
      t = items[0].type
      for x in items[1:]:
        t = join(t, x.type)
      elt = t.strong
    vals = [el.tmp(cast(x, elt)) for x in items]
    n = len(vals)
    if n <= 8:

      def fn(idx):
        out = vals[-1]
        for k in range(n - 2, -1, -1):
          out = ops.select(cmp("eq", idx[0], k), vals[k], out)
        return out

      arr = el.delayed(elt, (Const(n, I),), fn, cost=n)
    else:
      arr = el.alloc_array(elt, (Const(n, I),), label="lit")
      for k, v in enumerate(vals):
        el.emit(Store(arr.buf, Const(k, I), v))
    return arr
  arrays = []
  for x in items:
    if isinstance(x, Expr):
      x = el.scalar_to_array(x)
    if not isinstance(x, ArrayVal):
      raise el.error(TypeInferenceError, f"cannot build an array from {describe(x)}")
    arrays.append(x)
  rank = arrays[0].rank
  if any(a.rank != rank for a in arrays):
    raise el.error(
      TypeInferenceError,
      "cannot build an array from elements of different ranks: " + ", ".join(describe(a) for a in arrays),
    )
  return stack_arrays(el, arrays, 0, dtype)


def stack_arrays(el, arrays: list[ArrayVal], axis: int = 0, dtype: ScalarType | None = None) -> ArrayVal:
  """New array with ``arrays`` stacked along a new ``axis``."""
  elt = dtype
  if elt is None:
    elt = arrays[0].elt
    for a in arrays[1:]:
      elt = join(elt, a.elt)
  inner = arrays[0].shape
  for a in arrays[1:]:
    for d0, d in zip(inner, a.shape):
      if not el.dim_equal(d0, d):
        el.check(cmp("eq", d0, d), "ValueError", "all input arrays must have the same shape")
  rank = arrays[0].rank + 1
  axis = norm_axis(el, axis, rank)
  shape = inner[:axis] + (Const(len(arrays), I),) + inner[axis:]
  out = el.alloc_array(elt, shape, label="stack")
  for k, a in enumerate(arrays):
    if rank == 1:
      el.emit(Store(out.buf, Const(k, I), cast(el.elem(a, ()), elt)))
    else:
      el.store_array(el.axis_view(out, axis, Const(k, I)), a)
  if rank == 1:
    el.record_store(out)
  return out


def norm_axis(el, axis, rank: int, what: str = "axis") -> int:
  a = el.static_int(axis, what) if not isinstance(axis, int) else axis
  if not -rank <= a < rank:
    raise el.error(StaticValueError, f"{what} {a} is out of bounds for array of dimension {rank}")
  return a % rank


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------


def block_weight(block: list) -> int:
  """Rough cost of a scratch block: small only if it is straight-line scalars."""
  n = 0
  for s in walk_stmts(block):
    if not isinstance(s, (Assign, _Check, _Comment)):
      return INF_COST
    n += 1
  return n


def _defined_vars(block: list) -> set[int]:
  from ..ir import stmt_defs

  out: set[int] = set()
  for s in walk_stmts(block):
    out.update(v.uid for v in stmt_defs(s))
  return out


def outer_expr(el, e: Expr, inner: set[int], depth: int = 0) -> Expr | None:
  """Rewrite ``e`` so it no longer mentions variables defined in a scratch
  block (by substituting their pure definitions); None if impossible."""
  if isinstance(e, Const):
    return e
  if isinstance(e, Var):
    if e.uid not in inner:
      return e
    d = el.pure_defs.get(e.uid)
    if d is None or depth > 40:
      return None
    return outer_expr(el, d, inner, depth + 1)
  if isinstance(e, ops.Prim):
    args = [outer_expr(el, a, inner, depth + 1) for a in e.args]
    if any(a is None for a in args):
      return None
    return ops.prim(e.op, tuple(args), e.type)  # type: ignore[arg-type]
  if isinstance(e, ops.Cast):
    a = outer_expr(el, e.arg, inner, depth + 1)
    return None if a is None else cast(a, e.type)
  return None


def build_map(el, shape, fn: Callable[[tuple], Any], what: str = "map") -> ArrayVal:
  """Array whose element (or sub-array) at ``idx`` is ``fn(idx)``.

  ``fn`` is elaborated once in a scratch block to learn the result type; if
  it yields scalars the result stays delayed so it fuses with its consumer.
  """
  shape = tuple(el.tmp(d, "n") for d in shape)
  probe = tuple(Var("i", I) for _ in shape)
  for v, d in zip(probe, shape):
    el.nonneg.add(v.uid)
    el.upper[v.uid] = d
  # the result may be evaluated later, after variables have been rebound or
  # the calling function has returned: pin the lexical context
  context = el.capture()
  user_fn = fn

  def fn(idx):
    with el.restored(context):
      return user_fn(idx)

  el.loop_depth += 1
  try:
    with el.trial() as blk:
      r = fn(probe)
      if isinstance(r, tuple):
        raise el.error(
          UnsupportedError,
          f"the function passed to {what} returns a tuple; return a scalar or an array instead",
        )
      if isinstance(r, RangeVal):
        r = el.range_array(r)
      if not isinstance(r, (Expr, ArrayVal)):
        raise el.error(
          TypeInferenceError,
          f"the function passed to {what} must return a scalar or an array, got {describe(r)}",
        )
      weight = block_weight(blk)
      if isinstance(r, ArrayVal):
        inner_vars = _defined_vars(blk) | {v.uid for v in probe}
        inner_shape = [outer_expr(el, d, inner_vars) for d in r.shape]
        r_cost, r_elt, r_rank = r.cost, r.elt, r.rank
        r_concrete = r.buf is not None
      else:
        r_type = r.type.strong
  finally:
    el.loop_depth -= 1

  if isinstance(r, Expr):

    def at(idx):
      v = fn(tuple(idx))
      return cast(el.as_scalar(v, f"result of {what}"), r_type)

    return el.delayed(r_type, shape, at, roots=None, cost=min(weight + 1, INF_COST))

  lead = len(shape)
  if all(d is not None for d in inner_shape):
    inner_shape = tuple(el.tmp(d, "n") for d in inner_shape)
    full = shape + inner_shape
    if weight + r_cost <= CHEAP_COST:
      # pure and cheap: stay fully delayed
      def at(idx):
        sub = fn(tuple(idx[:lead]))
        return el.elem(sub, idx[lead:])

      return el.delayed(r_elt, full, at, roots=None, cost=weight + r_cost + (1 if r_concrete else 0))
    out = el.alloc_array(r_elt, full, label="out")

    def body(idx):
      sub = el.as_array(fn(tuple(idx)))
      for d, expect in zip(sub.shape, inner_shape):
        if not el.dim_equal(d, expect):
          el.check(
                        cmp("eq", d, expect), "ValueError",
                        f"the function passed to {what} returned arrays of different shapes ({{0}} vs {{1}})",
                        d, expect,
                    )  # fmt: skip
      el.store_array(sub_view(el, out, idx), sub)

    el.loop_nest(shape, body)
    return out

  # the shape of each result depends on the element: evaluate the first one to learn it
  total = el.tmp(ops.iprod(shape), "total")
  target = el.make_target(("a", r_elt, lead + r_rank), None, "out", el.scope)
  outer_blk = el.block
  some: list = []
  none: list = []
  with el.in_block(some):
    first = el.as_concrete(el.as_array(fn(el.unravel(ZERO, shape))))
    inner_shape = first.shape
    out = el.alloc_array(r_elt, shape + inner_shape, label="out")
    el.store_array(sub_view(el, out, el.unravel(ZERO, shape)), first)

    def body(idx):
      sub = el.as_array(fn(tuple(idx)))
      for d, expect in zip(sub.shape, inner_shape):
        el.check(
                    cmp("eq", d, expect), "ValueError",
                    f"the function passed to {what} returned arrays of different shapes ({{0}} vs {{1}})",
                    d, expect,
                )  # fmt: skip
      el.store_array(sub_view(el, out, idx), sub)

    if el.can_parallel(shape):
      el.par_loop(
        el.tmp(ops.isub(total, 1)),
        lambda c, start, stop: el.chunk_iter(shape, ops.iadd(start, 1), ops.iadd(stop, 1), body),
        base_depth=1 if len(shape) == 1 else 2,
      )
    else:
      el.chunk_iter(shape, ops.ONE, total, body)
    el.assign_target(target, out)
  with el.in_block(none):
    empty = el.alloc_array(r_elt, shape + (ZERO,) * r_rank, label="out")
    el.assign_target(target, empty)
  from ..ir import If

  outer_blk.append(If(cmp("gt", total, 0), some, none))
  return target


def sub_view(el, arr: ArrayVal, idx) -> ArrayVal:
  """``arr[idx]`` for a full leading index (a view of the trailing dims)."""
  lead = len(idx)
  mapping = [(None, i, ZERO) for i in idx] + [(k, ZERO, ops.ONE) for k in range(arr.rank - lead)]
  return el.remap(arr, arr.shape[lead:], mapping, consumed=True)


# ---------------------------------------------------------------------------
# reduce / scan
# ---------------------------------------------------------------------------


def _strip(e: Expr) -> Expr:
  while isinstance(e, ops.Cast):
    e = e.arg
  return e


def reduce_values(
  el,
  shape,
  elem: Callable[[tuple], Any],
  combine: Callable[[Any, Any], Any],
  init=None,
  what: str = "reduce",
  assume_associative: bool | None = None,
):
  """Fold ``combine`` over the scalars ``elem(idx)`` for all ``idx`` in ``shape``.

  Runs as a chunked parallel reduction when ``combine`` is a recognised
  associative primitive (or ``assume_associative`` says so).
  """
  shape = tuple(el.tmp(d, "n") for d in shape)
  probe = tuple(Var("i", I) for _ in shape)
  for v, d in zip(probe, shape):
    el.nonneg.add(v.uid)
    el.upper[v.uid] = d
  # ---- types: iterate acc = combine(acc, x) to a fixed point
  el.loop_depth += 1
  try:
    with el.trial():
      x0 = elem(probe)
      if not isinstance(x0, Expr):
        raise el.error(TypeInferenceError, f"{what} expects scalar elements here, got {describe(x0)}")
      x_type = x0.type
    acc_type = init.type if init is not None else x_type
    associative = False
    assoc_op = None
    for _ in range(6):
      a, x = Var("acc", acc_type), Var("x", x_type)
      with el.trial() as blk:
        r = combine(a, x)
        if not isinstance(r, Expr):
          raise el.error(
            TypeInferenceError,
            f"the function passed to {what} must return a scalar when given scalars, got {describe(r)}",
          )
        r_type = r.type
        core = _strip(r)
        associative = (
          not blk
          and isinstance(core, ops.Prim)
          and core.op in ops.ASSOCIATIVE
          and len(core.args) == 2
          and {id(_strip(core.args[0])), id(_strip(core.args[1]))} == {id(a), id(x)}
        )
        assoc_op = core.op if associative else None
      new = join(acc_type, r_type)
      if new is acc_type:
        break
      acc_type = new
    else:
      raise el.error(TypeInferenceError, f"the result type of the function passed to {what} does not stabilise")
  finally:
    el.loop_depth -= 1
  if assume_associative is not None:
    associative = assume_associative

  total = el.tmp(ops.iprod(shape), "total")
  acc = Var("acc", acc_type)
  el.mutables.add(acc.uid)
  empty_msg = f"zero-size array to {what}, which has no identity"

  def step(target: Var, x) -> None:
    el.emit(Assign(target, cast(el.as_scalar(combine(target, x), "reduction result"), acc_type)))

  def hint(target: Var):
    """Tell backends this loop is a reorderable fold (enables SIMD)."""
    return (assoc_op, target) if (associative and assoc_op in ("add", "mul")) else None

  if associative and el.can_parallel(shape):
    nchunks = el.tmp(RtCall("num_chunks", (total, Const(True, Bool)), I), "nchunks")
    el.nonneg.add(nchunks.uid) if isinstance(nchunks, Var) else None
    partials = el.alloc_array(acc_type.strong, (nchunks,), label="partials")

    def chunk_body(chunk, start, stop):
      local = Var("acc", acc_type)
      el.mutables.add(local.uid)
      el.emit(Assign(local, cast(el.as_scalar(elem(el.unravel(start, shape)), "element"), acc_type)))
      el.chunk_iter(shape, ops.iadd(start, 1), stop, lambda idx: step(local, elem(idx)), reduction=hint(local))
      el.emit(Store(partials.buf, chunk, cast(local, acc_type.strong)))

    el.par_loop(total, chunk_body, kind="reduce", nchunks=nchunks, base_depth=1 if len(shape) == 1 else 2)
    from ..ir import Load

    def partial(c):
      return cast(Load(partials.buf, c), acc_type)

    if init is not None:
      el.emit(Assign(acc, cast(init, acc_type)))
      el.for_range(ZERO, nchunks, lambda c: step(acc, partial(c)))
    else:
      el.check(cmp("gt", total, 0), "ValueError", empty_msg)
      el.emit(Assign(acc, partial(ZERO)))
      el.for_range(ops.ONE, nchunks, lambda c: step(acc, partial(c)))
  elif init is not None:
    el.emit(Assign(acc, cast(init, acc_type)))
    el.seq_nest(shape, lambda idx: step(acc, elem(idx)), reduction=hint(acc))
  else:
    el.check(cmp("gt", total, 0), "ValueError", empty_msg)
    el.emit(Assign(acc, cast(el.as_scalar(elem(el.unravel(ZERO, shape)), "element"), acc_type)))
    el.chunk_iter(shape, ops.ONE, total, lambda idx: step(acc, elem(idx)), reduction=hint(acc))
  return el.copy_scalar(acc, "red")


def reduce_arrays(el, n: Expr, elem: Callable[[Expr], Any], combine, init=None, what: str = "reduce"):
  """Sequential fold whose accumulator is an array (e.g. reducing over rows)."""
  n = el.tmp(n, "n")
  if init is None:
    el.check(cmp("gt", n, 0), "ValueError", f"zero-size array to {what}, which has no identity")
    first = elem(ZERO)
    start: Expr = ops.ONE
  else:
    first, start = init, ZERO
  # result type: fixed point of combine over value types
  vt = el.vtype(first)
  el.loop_depth += 1
  try:
    for _ in range(6):
      with el.trial():
        probe_acc = el.make_target(vt, first, "acc", el.scope, unknown_roots=True)
        r = combine(probe_acc, elem(Var("i", I)))
        new = el.join_vtypes(vt, el.vtype(r))
      if new is None:
        raise el.error(
          TypeInferenceError,
          f"the function passed to {what} returns {describe(r)}, which cannot be combined with "
          f"an accumulator of type {el.describe_vtype(vt)}",
        )
      if new == vt:
        break
      vt = new
  finally:
    el.loop_depth -= 1
  target = el.make_target(vt, first, "acc", el.scope, unknown_roots=True)
  el.assign_target(target, first)

  def body(i):
    r = combine(target, elem(i))
    el.assign_target(target, el.snapshot(r))

  el.for_range(start, n, body)
  return target


def scan_values(el, n: Expr, elem, combine, init=None, what: str = "scan") -> ArrayVal:
  """Running fold over a 1-d sequence of scalars; returns all partial results."""
  n = el.tmp(n, "n")
  probe = Var("i", I)
  el.nonneg.add(probe.uid)
  el.upper[probe.uid] = n
  el.loop_depth += 1
  try:
    with el.trial():
      x0 = elem(probe)
      if not isinstance(x0, Expr):
        raise el.error(UnsupportedError, f"{what} over sub-arrays is not supported; scan a 1-d array")
      x_type = x0.type
    acc_type = init.type if init is not None else x_type
    for _ in range(6):
      with el.trial():
        r = combine(Var("acc", acc_type), Var("x", x_type))
        if not isinstance(r, Expr):
          raise el.error(TypeInferenceError, f"the function passed to {what} must return a scalar")
        r_type = r.type
      new = join(acc_type, r_type)
      if new is acc_type:
        break
      acc_type = new
  finally:
    el.loop_depth -= 1
  out = el.alloc_array(acc_type.strong, (n,), label="scan")
  acc = Var("acc", acc_type)
  el.mutables.add(acc.uid)
  if init is not None:
    el.emit(Assign(acc, cast(init, acc_type)))

    def body(i):
      el.emit(Assign(acc, cast(el.as_scalar(combine(acc, elem(i)), "scan result"), acc_type)))
      el.emit(Store(out.buf, i, cast(acc, acc_type.strong)))

    el.for_range(ZERO, n, body)
  else:
    from ..ir import If

    first: list = []
    with el.in_block(first):
      el.emit(Assign(acc, cast(el.as_scalar(elem(ZERO), "element"), acc_type)))
      el.emit(Store(out.buf, ZERO, cast(acc, acc_type.strong)))
    el.emit(If(cmp("gt", n, 0), first, []))

    def body(i):
      el.emit(Assign(acc, cast(el.as_scalar(combine(acc, elem(i)), "scan result"), acc_type)))
      el.emit(Store(out.buf, i, cast(acc, acc_type.strong)))

    el.for_range(ops.ONE, n, body)
  return out


def along_axis(el, arr: ArrayVal, axis: int, fn: Callable[[ArrayVal], Any], what: str = "reduction") -> Any:
  """Apply ``fn`` to every 1-d slice of ``arr`` along ``axis`` (delayed)."""
  if arr.rank == 1:
    return fn(arr)
  if arr.buf is None and arr.cost > CHEAP_COST:
    arr = el.as_concrete(arr)
  el.consume(arr)
  lead = arr.shape[:axis] + arr.shape[axis + 1 :]

  def at(idx):
    mapping = []
    k2 = 0
    for k in range(arr.rank):
      if k == axis:
        mapping.append((0, ZERO, ops.ONE))
      else:
        mapping.append((None, idx[k2], ZERO))
        k2 += 1
    return fn(el.remap(arr, (arr.shape[axis],), mapping, consumed=True))

  return build_map(el, lead, at, what)


def is_scalar(v) -> bool:
  return isinstance(v, Expr)


def static_type(el, v) -> TypeVal | None:
  return v if isinstance(v, TypeVal) else None


__all__ = [n for n in dir() if not n.startswith("__")]
_ = (Int64, walk_expr)
