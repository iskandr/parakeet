"""Registry of library functions plus the core building blocks they share.

A *staged implementation* is a Python function ``impl(el, *args, **kwargs)``
that receives compile-time values and uses the elaborator ``el`` to build IR.
Implementations are registered against the Python objects they stand for
(``np.zeros``, ``len``, ...) so user code keeps calling the ordinary names.
"""

from __future__ import annotations

import contextlib
import difflib
import inspect
import types as pytypes
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import prims
from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..frontend import signature as _signature
from ..ndtypes import Bool, Float64, Int64, ScalarType, join
from ..syntax import ops
from ..syntax.ir import Assign, Const, Expr, Store, Var, walk_expr, walk_stmts
from ..syntax.ir import Check as _Check
from ..syntax.ir import Comment as _Comment
from ..syntax.ops import ZERO, I, cast, cmp
from ..syntax.values import CHEAP_COST, INF_COST, ArrayVal, PyFunc, RangeVal, TypeVal, describe

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
        args, kwargs = list(bound.args[1:]), {**bound.kwargs, **extra}
  el.lib_calls.append((name, len(el.frames)))
  try:
    return impl(el, *args, **kwargs)
  finally:
    el.lib_calls.pop()


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
      with packing(el, arr) as parts:
        for k, v in enumerate(vals):
          el.emit(Store(arr.buf, Const(k, I), v))
          parts.append((element_view(arr, k), v))
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
  jvp = el.jvp
  if jvp is not None and not jvp.muted and elt.kind == "f" and any(jvp.has_tangent(el, a) for a in arrays):
    jvp.ensure_zero_buf_tangent(el, out, label="d_stack")
  with packing(el, out) as parts:
    for k, a in enumerate(arrays):
      if rank == 1:
        val = el.tmp(cast(el.elem(a, ()), elt), "sv")
        el.emit(Store(out.buf, Const(k, I), val))
        if jvp is not None and not jvp.muted and out.buf.uid in jvp.buf_t:
          dv = jvp.tangent_expr(el, val)
          with jvp.computing(el):
            el.emit(Store(jvp.buf_t[out.buf.uid], Const(k, I), cast(dv, elt) if dv is not None else Const(0.0, elt)))
        parts.append((element_view(out, k), a))
      else:
        view = el.axis_view(out, axis, Const(k, I))
        el.store_array(view, a)
        parts.append((view, a))
    if rank == 1:
      el.record_store(out)
  return out


@contextlib.contextmanager
def packing(el, out: ArrayVal):
  """Assemble the new array ``out`` from parts.  The body appends
  ``(view of out, value stored there)`` pairs; under differentiation that is
  all the tape needs to route the adjoint of ``out`` back to the parts."""
  parts: list = []
  tape = el.tape
  if tape is None or tape.muted:
    yield parts
    return
  with tape.computing(el):
    yield parts
  tape.pack(el, parts, out)


def element_view(arr: ArrayVal, k: int) -> ArrayVal:
  """Element ``k`` of a 1-d array as a 0-d view."""
  return ArrayVal(arr.elt, (), buf=arr.buf, offset=ops.iadd(arr.offset, Const(k, I)), strides=(), roots=arr.roots)


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
  from ..syntax.ir import stmt_defs

  out: set[int] = set()
  for s in walk_stmts(block):
    out.update(v.uid for v in stmt_defs(s))
  return out


def outer_expr(el, e: Expr, inner: set[int]) -> Expr | None:
  """Rewrite ``e`` so it no longer mentions variables defined in ``inner``."""
  from ..transforms.lowering import outer_expr as _outer_expr

  return _outer_expr(el.pure_defs, e, inner)


def build_map(el, shape, fn: Callable[[tuple], Any], what: str = "map") -> ArrayVal:
  """Array whose element (or sub-array) at ``idx`` is ``fn(idx)``.

  ``fn`` is elaborated once in a scratch block to learn the result type; if
  it yields scalars the result stays delayed so it fuses with its consumer.
  """
  tape = el.tape
  if tape is not None and not tape.muted:
    out = tape.map(el, plain_map, shape, fn, what)
    if out is not None:
      return out
  return plain_map(el, shape, fn, what)


def plain_map(el, shape, fn: Callable[[tuple], Any], what: str = "map") -> ArrayVal:
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
  r_has_jvp = False
  try:
    with el.trial() as blk:
      r = fn(probe)
      if isinstance(r, tuple):
        if not r:
          raise el.error(UnsupportedError, f"the function passed to {what} returned an empty tuple")
        inner_vars = _defined_vars(blk) | {v.uid for v in probe}
        tuple_specs: list[tuple] = []
        for slot in r:
          if isinstance(slot, RangeVal):
            slot = el.range_array(slot)
          if isinstance(slot, Expr):
            tuple_specs.append(("s", slot.type.strong))
          elif isinstance(slot, ArrayVal):
            slot_inner = [outer_expr(el, d, inner_vars) for d in slot.shape]
            if any(d is None for d in slot_inner):
              raise el.error(
                UnsupportedError,
                f"multi-output {what} requires array outputs whose shapes do not vary per element",
              )
            tuple_specs.append(("a", slot.elt, tuple(slot_inner)))
          else:
            raise el.error(
              TypeInferenceError,
              f"tuple elements returned by {what} must be scalars or arrays, got {describe(slot)}",
            )
        is_tuple = True
      else:
        is_tuple = False
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
          if el.jvp is not None and not el.jvp.muted and r_elt.kind == "f":
            r_has_jvp = el.jvp.has_tangent_array(el, r)
        else:
          r_type = r.type.strong
  finally:
    el.loop_depth -= 1

  if is_tuple:
    outs: list[ArrayVal] = []
    pinned_specs: list[tuple] = []
    for k_slot, spec in enumerate(tuple_specs):
      if spec[0] == "s":
        outs.append(el.alloc_array(spec[1], shape, label=f"out{k_slot}"))
        pinned_specs.append(spec)
      else:
        in_sh = tuple(el.tmp(d, "n") for d in spec[2])
        outs.append(el.alloc_array(spec[1], shape + in_sh, label=f"out{k_slot}"))
        pinned_specs.append(("a", spec[1], in_sh))

    def tuple_body(idx):
      res = fn(tuple(idx))
      for out_j, spec_j, val_j in zip(outs, pinned_specs, res):
        if spec_j[0] == "s":
          pos = el.tmp(el.index_expr(out_j, idx), "pos")
          el.emit(Store(out_j.buf, pos, cast(el.as_scalar(val_j, f"result of {what}"), spec_j[1])))
        else:
          sub = el.as_array(val_j)
          for d, expect in zip(sub.shape, spec_j[2]):
            if not el.dim_equal(d, expect):
              el.check(
                cmp("eq", d, expect),
                "ValueError",
                f"the function passed to {what} returned arrays of different shapes ({{0}} vs {{1}})",
                d,
                expect,
              )
          el.store_array(sub_view(el, out_j, idx), sub)

    el.loop_nest(shape, tuple_body)
    return tuple(outs)

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
    if r_has_jvp and el.jvp is not None and not el.jvp.muted:
      el.jvp.ensure_zero_buf_tangent(el, out, label="d_out")

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

    rest = el.tmp(ops.isub(total, 1), "rest")
    el.index_map(
      (rest,), lambda flat: body(el.unravel(ops.iadd(flat[0], 1), shape)), ordered=not el.can_parallel(shape)
    )
    el.assign_target(target, out)
  with el.in_block(none):
    empty = el.alloc_array(r_elt, shape + (ZERO,) * r_rank, label="out")
    el.assign_target(target, empty)
  from ..syntax.ir import If

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


def _is_lossless_widening(src, dst) -> bool:
  """True when casting ``src -> dst`` preserves every value without narrowing or
  truthiness collapse (e.g. ``int64 -> bool`` is NOT value-preserving)."""
  if src.strong is dst.strong:
    return True
  if dst.kind == "b":
    return False
  if src.kind == "b":
    return True
  if src.kind == dst.kind:
    return dst.bits >= src.bits
  if src.kind == "u" and dst.kind == "i":
    return dst.bits > src.bits
  if src.kind in ("i", "u") and dst.kind == "f":
    return (dst.bits == 32 and src.bits <= 16) or (dst.bits == 64 and src.bits <= 32)
  return False


def _strip(e: Expr) -> Expr:
  while isinstance(e, ops.Cast) and _is_lossless_widening(e.arg.type, e.type):
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
  # the fold itself is an operator statement; what loops it becomes is up to the passes
  from ..syntax.ir import ErrorInfo, Fold, IndexReduce

  index = tuple(Var("i", I) for _ in shape)
  for v, d in zip(index, shape):
    el.nonneg.add(v.uid)
    el.upper[v.uid] = d
  threads = bool(associative and assoc_op is not None and el.can_parallel(shape))
  body: list = []
  scope = el.new_scope("loop")
  with el.in_block(body, scope, loop=True, par=threads):
    value = el.tmp(el.as_scalar(elem(index), "element"), "x")
  a, b = Var("acc", acc_type), Var("x", value.type)
  combine_block: list = []
  with el.in_block(combine_block, loop=True):
    combined = cast(el.as_scalar(combine(a, b), "reduction result"), acc_type)
  result = Var("red", acc_type)
  empty = (
    None if init is not None else ErrorInfo("ValueError", f"zero-size array to {what}, which has no identity", el.loc())
  )
  fold = Fold(
    result, value, a, b, combine_block, combined, None if init is None else cast(init, acc_type),
    assoc_op if associative else None, empty,
  )  # fmt: skip
  jvp = el.jvp
  if jvp is not None and not jvp.muted and acc_type.kind == "f":
    # Probe whether elem(probe) or init carries a tangent
    el.loop_depth += 1
    try:
      with el.trial():
        xp = el.as_scalar(elem(probe), "element")
        elem_active = jvp.has_tangent_expr(el, xp)
        init_active = init is not None and jvp.has_tangent_expr(el, init)
        # Also check if combine closes over a tangent variable
        if not (elem_active or init_active):
          r_probe = combine(Var("pa", acc_type), Var("px", x_type))
          combine_active = isinstance(r_probe, Expr) and jvp.has_tangent_expr(el, r_probe)
        else:
          combine_active = False
    finally:
      el.loop_depth -= 1
    if elem_active or init_active or combine_active:
      if associative and assoc_op == "add":
        # Linear add reduction: emit primal IndexReduce untouched (preserves monoid tiling!)
        # and a parallel tangent add reduction over d(elem(idx)).
        el.emit(IndexReduce(shape, index, body, [fold], scope, el.scope, threads, el.loc()))
        d_init = jvp.tangent_expr(el, init) if init is not None else Const(0.0, acc_type)
        d_init = Const(0.0, acc_type) if d_init is None else cast(d_init, acc_type)

        def d_elem(idx):
          xv = el.tmp(el.as_scalar(elem(idx), "element"), "x")
          dx = jvp.tangent_expr(el, xv)
          return Const(0.0, acc_type) if dx is None else cast(dx, acc_type)

        with jvp.computing(el):
          d_res = reduce_values(
            el,
            shape,
            d_elem,
            lambda u, v: ops.prim("add", (cast(u, acc_type), cast(v, acc_type)), acc_type),
            d_init,
            what,
            assume_associative=True,
          )
        jvp.scalar_t[result.uid] = cast(d_res, acc_type)
        return result
      if not (associative and assoc_op in ("min", "max", "mul")):
        # General arbitrary fold: single-pass O(1)-memory forward loop carrying (acc, d_acc).
        total = el.tmp(ops.iprod(shape), "total")
        if init is None:
          el.check(cmp("gt", total, 0), "ValueError", f"zero-size array to {what}, which has no identity")
        acc_v = Var("red", acc_type)
        dacc_v = Var("d_red", acc_type)
        el.mutables.add(acc_v.uid)
        el.mutables.add(dacc_v.uid)
        jvp.targets[acc_v.uid] = dacc_v
        if init is None:
          first_idx = el.unravel(ZERO, shape)
          x0v = el.tmp(el.as_scalar(elem(first_idx), "element"), "x0")
          dx0v = jvp.tangent_expr(el, x0v)
          el.emit(Assign(acc_v, cast(x0v, acc_type)))
          with jvp.computing(el):
            el.emit(Assign(dacc_v, cast(dx0v, acc_type) if dx0v is not None else Const(0.0, acc_type)))
          start_i = ops.ONE
        else:
          d_init = jvp.tangent_expr(el, init)
          el.emit(Assign(acc_v, cast(init, acc_type)))
          with jvp.computing(el):
            el.emit(Assign(dacc_v, cast(d_init, acc_type) if d_init is not None else Const(0.0, acc_type)))
          start_i = ZERO

        def fold_step(flat_i: Expr) -> None:
          idx = el.unravel(flat_i, shape)
          xv = el.tmp(el.as_scalar(elem(idx), "element"), "x")
          rv = el.tmp(cast(el.as_scalar(combine(acc_v, xv), "reduction result"), acc_type), "r")
          drv = jvp.tangent_expr(el, rv)
          el.emit(Assign(acc_v, rv))
          with jvp.computing(el):
            el.emit(Assign(dacc_v, cast(drv, acc_type) if drv is not None else Const(0.0, acc_type)))

        el.for_range(start_i, total, fold_step, name="fi")
        jvp.scalar_t[acc_v.uid] = dacc_v
        return acc_v
  el.emit(IndexReduce(shape, index, body, [fold], scope, el.scope, threads, el.loc()))
  return result


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
  jvp = el.jvp
  if jvp is not None and not jvp.muted:
    jvp.make_tangent_target(el, target, "acc", el.scope)
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
  jvp = el.jvp
  if jvp is not None and not jvp.muted and acc_type.kind == "f":
    el.loop_depth += 1
    try:
      with el.trial():
        xp = el.as_scalar(elem(probe), "element")
        elem_active = jvp.has_tangent_expr(el, xp)
        init_active = init is not None and jvp.has_tangent_expr(el, init)
        if not (elem_active or init_active):
          r_probe = combine(Var("pa", acc_type), Var("px", x_type))
          combine_active = isinstance(r_probe, Expr) and jvp.has_tangent_expr(el, r_probe)
        else:
          combine_active = False
    finally:
      el.loop_depth -= 1
    if elem_active or init_active or combine_active:
      d_out = jvp.ensure_zero_buf_tangent(el, out, label="d_scan")
      acc_v = Var("run", acc_type)
      dacc_v = Var("d_run", acc_type)
      el.mutables.add(acc_v.uid)
      el.mutables.add(dacc_v.uid)
      jvp.targets[acc_v.uid] = dacc_v
      if init is not None:
        d_init = jvp.tangent_expr(el, init)
        el.emit(Assign(acc_v, cast(init, acc_type)))
        with jvp.computing(el):
          el.emit(Assign(dacc_v, cast(d_init, acc_type) if d_init is not None else Const(0.0, acc_type)))
      else:
        el.emit(Assign(acc_v, Const(0.0, acc_type)))
        with jvp.computing(el):
          el.emit(Assign(dacc_v, Const(0.0, acc_type)))

      def scan_step(i: Expr) -> None:
        xv = el.tmp(el.as_scalar(elem(i), "element"), "x")
        if init is None:
          first_blk: list = []
          rest_blk: list = []
          with el.in_block(first_blk, loop=True):
            dx0 = jvp.tangent_expr(el, xv)
            el.emit(Assign(acc_v, cast(xv, acc_type)))
            with jvp.computing(el):
              el.emit(Assign(dacc_v, cast(dx0, acc_type) if dx0 is not None else Const(0.0, acc_type)))
          with el.in_block(rest_blk, loop=True):
            rv = el.tmp(cast(el.as_scalar(combine(acc_v, xv), "scan result"), acc_type), "r")
            drv = jvp.tangent_expr(el, rv)
            el.emit(Assign(acc_v, rv))
            with jvp.computing(el):
              el.emit(Assign(dacc_v, cast(drv, acc_type) if drv is not None else Const(0.0, acc_type)))
          from ..syntax.ir import If

          el.emit(If(cmp("eq", i, ZERO), first_blk, rest_blk))
        else:
          rv = el.tmp(cast(el.as_scalar(combine(acc_v, xv), "scan result"), acc_type), "r")
          drv = jvp.tangent_expr(el, rv)
          el.emit(Assign(acc_v, rv))
          with jvp.computing(el):
            el.emit(Assign(dacc_v, cast(drv, acc_type) if drv is not None else Const(0.0, acc_type)))
        el.emit(Store(out.buf, i, cast(acc_v, acc_type.strong)))
        with jvp.computing(el):
          el.emit(Store(d_out.buf, i, cast(dacc_v, acc_type.strong)))

      el.for_range(ZERO, n, scan_step, name="si")
      el.record_store(out)
      return out
  # the scan itself is an operator statement; what loops it becomes is up to the passes
  from ..syntax.ir import Fold, IndexScan

  threads = _is_associative(el, combine, acc_type, x_type) and el.can_parallel((n,))
  index = Var("i", I)
  el.nonneg.add(index.uid)
  el.upper[index.uid] = n
  body: list = []
  scope = el.new_scope("loop")
  with el.in_block(body, scope, loop=True, par=threads):
    value = el.tmp(el.as_scalar(elem(index), "element"), "x")
  a, b = Var("acc", acc_type), Var("x", value.type)
  combine_block: list = []
  with el.in_block(combine_block, loop=True):
    combined = cast(el.as_scalar(combine(a, b), "scan result"), acc_type)
  result = Var("run", acc_type)
  fold = Fold(result, value, a, b, combine_block, combined, None if init is None else cast(init, acc_type))
  el.emit(
    IndexScan(n, index, body, fold, [Store(out.buf, index, cast(result, acc_type.strong))], scope, threads, el.loc())
  )
  return out


def _is_associative(el, combine, acc_type, x_type) -> bool:
  """Is ``combine`` one of the primitives whose folds may be regrouped?"""
  a, x = Var("acc", acc_type), Var("x", x_type)
  el.loop_depth += 1
  try:
    with el.trial() as blk:
      r = combine(a, x)
      core = _strip(r) if isinstance(r, Expr) else None
      return (
        not blk
        and isinstance(core, ops.Prim)
        and core.op in ops.ASSOCIATIVE
        and len(core.args) == 2
        and {id(_strip(core.args[0])), id(_strip(core.args[1]))} == {id(a), id(x)}
      )
  finally:
    el.loop_depth -= 1


def along_axis(
  el, arr: ArrayVal, axis: int, fn: Callable[[ArrayVal], Any], what: str = "reduction", *, stream=False
) -> Any:
  """Apply ``fn`` to every 1-d slice of ``arr`` along ``axis`` (delayed)."""
  if arr.rank == 1:
    return fn(arr)
  if not stream and arr.buf is None and arr.cost > CHEAP_COST:
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
