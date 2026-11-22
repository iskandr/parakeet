"""Staged implementations of the supported NumPy API."""

from __future__ import annotations

import numpy as np

from .. import prims
from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..ndtypes import Bool, Float32, Float64, Int64, ScalarType, UInt64, WeakBool, join, weak_of_kind
from ..syntax import ops
from ..syntax.ir import Assign, Const, Expr, If, Load, RtCall, Store, Var
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, imul, isub
from ..syntax.values import CHEAP_COST, ArrayVal, RangeVal, TypeVal, describe
from .core import (
  along_axis,
  array_literal,
  build_map,
  convert,
  method,
  norm_axis,
  packing,
  prop,
  reduce_values,
  register,
  stack_arrays,
)

UF = prims.UFUNCS


def _arr(el, v, what: str = "argument") -> ArrayVal:
  """Coerce to an array (scalars become 0-d arrays)."""
  return el.as_array(v, what)


def _dtype(el, dtype, default: ScalarType) -> ScalarType:
  return default if dtype is None else el.as_dtype(dtype)


def _float_type(elt: ScalarType) -> ScalarType:
  return elt if elt.kind == "f" else Float64


def _use(el, a: ArrayVal) -> ArrayVal:
  if a.buf is None and a.cost > CHEAP_COST:
    a = el.as_concrete(a)
  el.consume(a)
  return a


def _axes(el, axis, rank: int) -> list[int] | None:
  if axis is None:
    return None
  if isinstance(axis, tuple):
    out = [norm_axis(el, a, rank) for a in axis]
    if len(set(out)) != len(out):
      raise el.error(StaticValueError, "duplicate value in 'axis'")
    return sorted(out)
  return [norm_axis(el, axis, rank)]


def _keepdims(el, result, arr: ArrayVal, axes, keepdims):
  if not el.static_bool(keepdims, "keepdims") if not isinstance(keepdims, bool) else not keepdims:
    return result
  res = el.as_array(result)
  positions = list(range(arr.rank)) if axes is None else axes
  return el.expand_dims(res, positions)


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


@prop("shape")
def _shape(el, x):
  return x.shape if isinstance(x, ArrayVal) else ()


@prop("ndim")
def _ndim(el, x):
  return Const(x.rank if isinstance(x, ArrayVal) else 0, I)


@prop("size")
def _size(el, x):
  return el.tmp(ops.iprod(x.shape), "size") if isinstance(x, ArrayVal) else ONE


@prop("dtype")
def _dtype_prop(el, x):
  return TypeVal(x.elt if isinstance(x, ArrayVal) else x.type.strong)


@prop("itemsize")
def _itemsize(el, x):
  return Const((x.elt if isinstance(x, ArrayVal) else x.type).itemsize, I)


@prop("nbytes")
def _nbytes(el, x):
  return el.tmp(imul(_size(el, x), _itemsize(el, x)), "nbytes")


@prop("T")
def _T(el, x):
  return el.transpose(x) if isinstance(x, ArrayVal) else x


@prop("real")
def _real(el, x):
  return x


@prop("imag")
def _imag(el, x):
  if isinstance(x, ArrayVal):
    return el.delayed(x.elt, x.shape, lambda idx: Const(0, x.elt), cost=0)
  return Const(0, x.type)


@prop("flat")
def _flat(el, x):
  return el.flat_view(_arr(el, x))


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------


def _full(el, shape, value: Expr | None, elt: ScalarType, zero: bool = False) -> ArrayVal:
  dims = el.shape_tuple(shape)
  out = el.alloc_array(elt, dims, zero=zero)
  if value is not None:
    el.fill(out, value)
  return out


@register(np.zeros)
def zeros(el, shape, dtype=None, order="C"):
  return _full(el, shape, None, _dtype(el, dtype, Float64), zero=True)


@register(np.ones)
def ones(el, shape, dtype=None, order="C"):
  elt = _dtype(el, dtype, Float64)
  return _full(el, shape, Const(1, elt), elt)


@register(np.empty)
def empty(el, shape, dtype=None, order="C"):
  return _full(el, shape, None, _dtype(el, dtype, Float64))


@register(np.full)
def full(el, shape, fill_value, dtype=None, order="C"):
  value = el.as_scalar(fill_value, "fill_value")
  elt = _dtype(el, dtype, value.type.strong)
  return _full(el, shape, cast(value, elt), elt)


def _like(el, a, dtype):
  a = _arr(el, a)
  return a, _dtype(el, dtype, a.elt)


@register(np.zeros_like)
def zeros_like(el, a, dtype=None):
  a, elt = _like(el, a, dtype)
  return _full(el, a.shape, None, elt, zero=True)


@register(np.ones_like)
def ones_like(el, a, dtype=None):
  a, elt = _like(el, a, dtype)
  return _full(el, a.shape, Const(1, elt), elt)


@register(np.empty_like)
def empty_like(el, a, dtype=None):
  a, elt = _like(el, a, dtype)
  return _full(el, a.shape, None, elt)


@register(np.full_like)
def full_like(el, a, fill_value, dtype=None):
  a, elt = _like(el, a, dtype)
  return _full(el, a.shape, cast(el.as_scalar(fill_value, "fill_value"), elt), elt)


@register(np.arange)
def arange(el, start, stop=None, step=None, dtype=None):
  if stop is None:
    start, stop = Const(0, I), start
  if step is None:
    step = Const(1, I)
  parts = [el.as_scalar(v, "arange argument") for v in (start, stop, step)]
  t = parts[0].type
  for p in parts[1:]:
    t = join(t, p.type)
  if t.kind == "f":
    ft = t.strong if dtype is None else el.as_dtype(dtype)
    start_f, stop_f, step_f = (el.tmp(cast(p, Float64)) for p in parts)
    el.check(ops.prim("ne", (step_f, Const(0.0, Float64)), Bool), "ZeroDivisionError", "arange step cannot be zero")
    span = ops.prim("div", (ops.prim("sub", (stop_f, start_f), Float64), step_f), Float64)
    n = el.tmp(ops.imax(cast(ops.prim("ceil", (span,), Float64), I), 0), "n")
    el.nonneg.add(n.uid) if isinstance(n, Var) else None

    def fn(idx):
      x = ops.prim("add", (start_f, ops.prim("mul", (cast(idx[0], Float64), step_f), Float64)), Float64)
      return cast(x, ft)

    return el.delayed(ft, (n,), fn, cost=2)
  rng = RangeVal(cast(parts[0], I), cast(parts[1], I), cast(parts[2], I))
  arr = el.range_array(rng)
  elt = Int64 if t.weak or t.kind == "b" else t.strong
  if dtype is not None:
    elt = el.as_dtype(dtype)
  return el.cast_array(arr, elt)


@register(np.eye)
def eye(el, N, M=None, k=0, dtype=None):
  n = el.as_index(N, "N")
  m = n if M is None else el.as_index(M, "M")
  kk = el.tmp(el.as_index(k, "k"))
  elt = _dtype(el, dtype, Float64)
  n, m = el.check_dims((n, m))
  return el.delayed(
        elt, (n, m),
        lambda idx: ops.select(cmp("eq", isub(idx[1], idx[0]), kk), Const(1, elt), Const(0, elt)),
        cost=1,
    )  # fmt: skip


@register(np.identity)
def identity(el, n, dtype=None):
  return eye(el, n, dtype=dtype)


@register(np.array, np.asarray, np.ascontiguousarray, np.asanyarray)
def array(el, obj, dtype=None, copy=None, order=None, ndmin=0):
  elt = None if dtype is None else el.as_dtype(dtype)
  if isinstance(obj, tuple):
    arr = array_literal(el, list(obj), elt)
  elif isinstance(obj, Expr):
    arr = el.scalar_to_array(obj if elt is None else cast(obj, elt))
  elif isinstance(obj, RangeVal):
    arr = el.range_array(obj)
  elif isinstance(obj, ArrayVal):
    arr = obj
  else:
    raise el.error(TypeInferenceError, f"cannot build an array from {describe(obj)}")
  if elt is not None:
    arr = el.cast_array(arr, elt)
  return arr


@register(np.copy)
@method("copy")
def copy(el, a, order="C"):
  a = _arr(el, a)
  return el.materialize(a)


@method("astype")
def astype(el, a, dtype, copy=True):
  a = _arr(el, a)
  elt = el.as_dtype(dtype)
  if a.elt is elt and a.buf is not None:
    return el.materialize(a)
  return el.cast_array(a, elt)


@method("fill")
def fill(el, a, value):
  if not isinstance(a, ArrayVal):
    raise el.error(TypeInferenceError, "fill() needs an array")
  el.force_inplace(a)
  el.store_array(a, el.as_scalar(value, "fill value"))
  return None


@method("item")
def item(el, a, *index):
  a = _arr(el, a)
  if len(index) == 1 and not isinstance(index[0], tuple):
    v = el.as_scalar(el.getitem(el.flat_view(a) if a.rank != 1 else a, index[0]))
  elif index:
    v = el.as_scalar(el.getitem(a, tuple(index) if len(index) > 1 else index[0]))
  else:
    v = el.as_scalar(a, "array passed to item()")
  return cast(v, weak_of_kind(v.type.kind))  # item() returns a Python scalar


@method("tolist")
def tolist(el, a):
  raise el.error(UnsupportedError, "tolist() is not supported: compiled code has no Python lists")


# ---------------------------------------------------------------------------
# shape manipulation
# ---------------------------------------------------------------------------


@register(np.reshape)
def reshape(el, a, shape=None, *more, order="C", newshape=None):
  if shape is None:
    shape = newshape
  if more:
    shape = (shape, *more)
  return el.reshape(_arr(el, a), el.shape_tuple(shape, "new shape"))


@method("reshape")
def _reshape_method(el, a, *shape, order="C"):
  if len(shape) == 1:
    shape = shape[0]
  return el.reshape(_arr(el, a), el.shape_tuple(shape, "new shape"))


@register(np.ravel)
@method("ravel")
def ravel(el, a, order="C"):
  return el.flat_view(_arr(el, a))


@method("flatten")
def flatten(el, a, order="C"):
  return el.materialize(el.flat_view(_arr(el, a)))


@register(np.transpose)
def transpose(el, a, axes=None):
  a = _arr(el, a)
  if axes is None:
    return el.transpose(a)
  perm = [el.static_int(x, "axes") % max(a.rank, 1) for x in (axes if isinstance(axes, tuple) else (axes,))]
  return el.transpose(a, perm)


@method("transpose")
def _transpose_method(el, a, *axes):
  if not axes:
    return transpose(el, a)
  if len(axes) == 1:
    return transpose(el, a, axes[0])
  return transpose(el, a, tuple(axes))


@register(np.swapaxes)
@method("swapaxes")
def swapaxes(el, a, axis1, axis2):
  a = _arr(el, a)
  i, j = norm_axis(el, axis1, a.rank), norm_axis(el, axis2, a.rank)
  perm = list(range(a.rank))
  perm[i], perm[j] = perm[j], perm[i]
  return el.transpose(a, perm)


@register(np.moveaxis)
def moveaxis(el, a, source, destination):
  a = _arr(el, a)
  s, d = norm_axis(el, source, a.rank), norm_axis(el, destination, a.rank)
  perm = [k for k in range(a.rank) if k != s]
  perm.insert(d, s)
  return el.transpose(a, perm)


@register(np.expand_dims)
def expand_dims(el, a, axis):
  a = _arr(el, a)
  axes = axis if isinstance(axis, tuple) else (axis,)
  rank = a.rank + len(axes)
  return el.expand_dims(a, [norm_axis(el, x, rank) for x in axes])


@register(np.squeeze)
@method("squeeze")
def squeeze(el, a, axis=None):
  a = _arr(el, a)
  if axis is None:
    drop = []
    for k, d in enumerate(a.shape):
      if isinstance(d, Const) and d.value == 1:
        drop.append(k)
      elif not el.is_not_one(d):
        raise el.error(
          StaticValueError,
          f"squeeze() cannot tell at compile time whether dimension {k} has length 1 "
          "(the number of dimensions of the result must be known)",
          hint="pass the axis explicitly, e.g. np.squeeze(x, axis=1)",
        )
  else:
    drop = _axes(el, axis, a.rank) or []
    for k in drop:
      d = a.shape[k]
      if isinstance(d, Const):
        if d.value != 1:
          raise el.error(StaticValueError, "cannot select an axis to squeeze out which has size not equal to one")
      else:
        el.check(
          cmp("eq", d, 1),
          "ValueError",
          "cannot select an axis to squeeze out which has size not equal to one",
        )
  if not drop:
    return a
  mapping = []
  shape = []
  for k in range(a.rank):
    if k in drop:
      mapping.append((None, ZERO, ZERO))
    else:
      mapping.append((len(shape), ZERO, ONE))
      shape.append(a.shape[k])
  return el.remap(a, shape, mapping)


@register(np.atleast_1d)
def atleast_1d(el, a):
  a = _arr(el, a)
  return el.expand_dims(a, [0]) if a.rank == 0 else a


@register(np.atleast_2d)
def atleast_2d(el, a):
  a = _arr(el, a)
  while a.rank < 2:
    a = el.expand_dims(a, [0])
  return a


@register(np.broadcast_to)
def broadcast_to(el, a, shape):
  a = _arr(el, a)
  return el.broadcast_to(a, el.check_dims(el.shape_tuple(shape)), check=True)


@register(np.flip)
def flip(el, a, axis=None):
  a = _arr(el, a)
  axes = _axes(el, axis, a.rank)
  if axes is None:
    axes = list(range(a.rank))
  mapping = []
  for k in range(a.rank):
    if k in axes:
      mapping.append((k, isub(a.shape[k], 1), Const(-1, I)))
    else:
      mapping.append((k, ZERO, ONE))
  return el.remap(a, a.shape, mapping)


@register(np.flipud)
def flipud(el, a):
  return flip(el, a, 0)


@register(np.fliplr)
def fliplr(el, a):
  return flip(el, a, 1)


def _sequence(el, seq, what: str) -> list[ArrayVal]:
  if isinstance(seq, tuple):
    return [_arr(el, x) for x in seq]
  if isinstance(seq, ArrayVal) and seq.rank >= 1 and isinstance(seq.shape[0], Const):
    return [el.as_array(el.getitem(seq, Const(k, I))) for k in range(seq.shape[0].value)]
  raise el.error(
    TypeInferenceError,
    f"{what} expects a tuple or list of arrays whose length is known at compile time, got {describe(seq)}",
  )


@register(np.concatenate)
def concatenate(el, arrays, axis=0, dtype=None):
  arrs = _sequence(el, arrays, "concatenate()")
  if not arrs:
    raise el.error(StaticValueError, "need at least one array to concatenate")
  if axis is None:
    arrs = [el.flat_view(a) for a in arrs]
    axis = 0
  rank = arrs[0].rank
  if rank == 0:
    raise el.error(TypeInferenceError, "zero-dimensional arrays cannot be concatenated")
  if any(a.rank != rank for a in arrs):
    raise el.error(TypeInferenceError, "all the input arrays must have the same number of dimensions")
  k = norm_axis(el, axis, rank)
  elt = arrs[0].elt
  for a in arrs[1:]:
    elt = join(elt, a.elt)
  if dtype is not None:
    elt = el.as_dtype(dtype)
  first = arrs[0]
  total: Expr = ZERO
  for a in arrs:
    total = iadd(total, a.shape[k])
    for d in range(rank):
      if d != k and not el.dim_equal(a.shape[d], first.shape[d]):
        el.check(
                    cmp("eq", a.shape[d], first.shape[d]), "ValueError",
                    f"all the input array dimensions except for the concatenation axis must match exactly "
                    f"(axis {d}: {{0}} vs {{1}})", first.shape[d], a.shape[d],
                )  # fmt: skip
  shape = first.shape[:k] + (el.tmp(total, "n"),) + first.shape[k + 1 :]
  out = el.alloc_array(elt, shape, label="cat")
  pos: Expr = ZERO
  with packing(el, out) as parts:
    for a in arrs:
      mapping = [(d, pos if d == k else ZERO, ONE) for d in range(rank)]
      view = el.remap(out, a.shape, mapping)
      el.store_array(view, a)
      parts.append((view, a))
      pos = el.tmp(iadd(pos, a.shape[k]), "pos")
  return out


@register(np.stack)
def stack(el, arrays, axis=0):
  arrs = _sequence(el, arrays, "stack()")
  if not arrs:
    raise el.error(StaticValueError, "need at least one array to stack")
  if any(a.rank != arrs[0].rank for a in arrs):
    raise el.error(TypeInferenceError, "all input arrays must have the same shape")
  return stack_arrays(el, arrs, el.static_int(axis, "axis") if not isinstance(axis, int) else axis)


@register(np.vstack)
def vstack(el, arrays):
  arrs = [atleast_2d(el, a) for a in _sequence(el, arrays, "vstack()")]
  return concatenate(el, tuple(arrs), 0)


@register(np.hstack)
def hstack(el, arrays):
  arrs = [atleast_1d(el, a) for a in _sequence(el, arrays, "hstack()")]
  return concatenate(el, tuple(arrs), 0 if arrs and arrs[0].rank == 1 else 1)


@register(np.column_stack)
def column_stack(el, arrays):
  arrs = []
  for a in _sequence(el, arrays, "column_stack()"):
    arrs.append(el.expand_dims(a, [1]) if a.rank < 2 else a)
  return concatenate(el, tuple(arrs), 1)


@register(np.tile)
def tile(el, a, reps):
  a = _arr(el, a)
  reps_t = el.check_dims(el.shape_tuple(reps, "reps"))
  rank = max(a.rank, len(reps_t))
  while a.rank < rank:
    a = el.expand_dims(a, [0])
  reps_t = (ONE,) * (rank - len(reps_t)) + tuple(reps_t)
  a = _use(el, a)
  shape = tuple(el.tmp(imul(d, r), "n") for d, r in zip(a.shape, reps_t))
  src_shape = a.shape

  def fn(idx):
    return el.elem(a, tuple(ops.imod(i, ops.imax(d, 1)) for i, d in zip(idx, src_shape)))

  return el.delayed(a.elt, shape, fn, roots=el.get_roots(a), cost=a.cost + rank)


@register(np.repeat)
def repeat(el, a, repeats, axis=None):
  a = _arr(el, a)
  r = el.tmp(el.as_index(repeats, "repeats"))
  el.check(cmp("ge", r, 0), "ValueError", "repeats may not contain negative values")
  if axis is None:
    a = el.flat_view(a) if a.rank != 1 else a
    k = 0
  else:
    k = norm_axis(el, axis, a.rank)
  a = _use(el, a)
  shape = tuple(el.tmp(imul(d, r), "n") if j == k else d for j, d in enumerate(a.shape))

  def fn(idx):
    return el.elem(a, tuple(ops.idiv(i, ops.imax(r, 1)) if j == k else i for j, i in enumerate(idx)))

  return el.delayed(a.elt, shape, fn, roots=el.get_roots(a), cost=a.cost + 1)


@register(np.diag)
def diag(el, v, k=0):
  v = _arr(el, v)
  kk = el.static_int(k, "k") if not isinstance(k, int) else k
  if v.rank == 1:
    v = _use(el, v)
    length = v.shape[0]
    n = el.tmp(iadd(length, abs(kk)), "n")
    zero = Const(0, v.elt)
    row_off = max(-kk, 0)

    def fn(idx):
      i, j = idx
      on_diag = cmp("eq", isub(j, i), kk)
      pos = ops.imin(ops.imax(isub(i, row_off), 0), ops.imax(isub(length, 1), 0))
      return ops.select(on_diag, el.elem(v, (pos,)), zero)

    return el.delayed(v.elt, (n, n), fn, roots=el.get_roots(v), cost=v.cost + 3)
  if v.rank == 2:
    rows, cols = v.shape
    row_off, col_off = max(-kk, 0), max(kk, 0)
    n = el.tmp(ops.imax(ops.imin(isub(rows, row_off), isub(cols, col_off)), 0), "n")
    el.nonneg.add(n.uid) if isinstance(n, Var) else None
    return el.remap(v, (n,), [(0, Const(row_off, I), ONE), (0, Const(col_off, I), ONE)])
  raise el.error(TypeInferenceError, "diag() input must be 1- or 2-d")


@register(np.diagonal)
@method("diagonal")
def diagonal(el, a, offset=0):
  a = _arr(el, a)
  if a.rank != 2:
    raise el.error(UnsupportedError, "diagonal() is only supported for 2-d arrays")
  return diag(el, a, offset)


@register(np.trace)
@method("trace")
def trace(el, a, offset=0):
  return sum_(el, diagonal(el, a, offset))


@register(np.tril)
def tril(el, m, k=0):
  return _tri(el, m, k, lower=True)


@register(np.triu)
def triu(el, m, k=0):
  return _tri(el, m, k, lower=False)


def _tri(el, m, k, lower: bool):
  m = _arr(el, m)
  if m.rank < 2:
    raise el.error(TypeInferenceError, "tril()/triu() need an array with at least two dimensions")
  kk = el.tmp(el.as_index(k, "k"))
  m = _use(el, m)
  zero = Const(0, m.elt)

  def fn(idx):
    d = isub(idx[-1], idx[-2])
    keep = cmp("le", d, kk) if lower else cmp("ge", d, kk)
    return ops.select(keep, el.elem(m, idx), zero)

  return el.delayed(m.elt, m.shape, fn, roots=el.get_roots(m), cost=m.cost + 2)


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


def _sum_type(elt: ScalarType) -> ScalarType:
  if elt.kind in "bi" and elt.bits < 64:
    return Int64
  if elt.kind == "u" and elt.bits < 64:
    return UInt64
  return elt


def _reduce_prim(el, op: str, a, axis=None, init: Expr | None = None, what: str = "", keepdims=False, elt=None):
  a = _arr(el, a)
  if elt is not None:
    a = el.cast_array(a, elt)
  tape = el.tape
  if tape is not None and not tape.muted and a.rank > 0 and op in ("add", "mul", "min", "max"):
    if tape.is_active(el, a):
      return tape.reduce(el, reduce_plain, op, a, axis, init, what, keepdims)
  return reduce_plain(el, op, a, axis, init, what, keepdims)


def reduce_plain(el, op: str, a: ArrayVal, axis, init, what: str, keepdims):
  """Reduce ``a`` with the primitive ``op`` over ``axis`` (None: everything)."""
  spec = UF[op]

  def combine(x, y):
    return el.apply_ufunc(spec, [x, y])

  def over(arr):
    arr = _use(el, arr)
    return reduce_values(el, arr.shape, lambda idx: el.tmp(el.elem(arr, idx), "x"), combine, init, what)

  axes = _axes(el, axis, a.rank)
  if a.rank == 0:
    return el.as_scalar(a)
  if axes is None or len(axes) == a.rank:
    res = over(a)
  elif len(axes) == 1:
    res = along_axis(el, a, axes[0], over, what)
  else:
    # several axes at once: move them to the end and reduce each trailing block in one pass
    from .core import sub_view

    kept = [k for k in range(a.rank) if k not in axes]
    arr = el.transpose(a, kept + axes)
    if arr.buf is None and arr.cost > CHEAP_COST:
      arr = el.as_concrete(arr)
    el.consume(arr)
    res = build_map(el, arr.shape[: len(kept)], lambda idx: over(sub_view(el, arr, idx)), what)
  return _keepdims(el, res, a, axes, keepdims)


@register(np.sum)
@method("sum")
def sum_(el, a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
  a = _arr(el, a)
  elt = el.as_dtype(dtype) if dtype is not None else _sum_type(a.elt)
  init = Const(0, elt) if initial is None else cast(el.as_scalar(initial, "initial"), elt)
  return _reduce_prim(el, "add", a, axis, init, "sum", keepdims, elt)


@register(np.prod)
@method("prod")
def prod(el, a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
  a = _arr(el, a)
  elt = el.as_dtype(dtype) if dtype is not None else _sum_type(a.elt)
  init = Const(1, elt) if initial is None else cast(el.as_scalar(initial, "initial"), elt)
  return _reduce_prim(el, "mul", a, axis, init, "prod", keepdims, elt)


@register(np.min, np.amin)
@method("min")
def amin(el, a, axis=None, out=None, keepdims=False, initial=None):
  init = None if initial is None else el.as_scalar(initial, "initial")
  return _reduce_prim(el, "min", a, axis, init, "minimum", keepdims)


@register(np.max, np.amax)
@method("max")
def amax(el, a, axis=None, out=None, keepdims=False, initial=None):
  init = None if initial is None else el.as_scalar(initial, "initial")
  return _reduce_prim(el, "max", a, axis, init, "maximum", keepdims)


@register(np.any)
@method("any")
def any_(el, a, axis=None, out=None, keepdims=False):
  a = el.cast_array(_arr(el, a), Bool)
  return _reduce_prim(el, "logical_or", a, axis, Const(False, Bool), "any", keepdims)


@register(np.all)
@method("all")
def all_(el, a, axis=None, out=None, keepdims=False):
  a = el.cast_array(_arr(el, a), Bool)
  return _reduce_prim(el, "logical_and", a, axis, Const(True, Bool), "all", keepdims)


@register(np.count_nonzero)
def count_nonzero(el, a, axis=None, keepdims=False):
  a = el.cast_array(el.cast_array(_arr(el, a), Bool), Int64)
  return _reduce_prim(el, "add", a, axis, Const(0, Int64), "count_nonzero", keepdims)


def _count(el, a: ArrayVal, axes) -> Expr:
  dims = a.shape if axes is None else [a.shape[k] for k in axes]
  return el.tmp(ops.iprod(dims), "count")


@register(np.mean)
@method("mean")
def mean(el, a, axis=None, dtype=None, out=None, keepdims=False):
  a = _arr(el, a)
  ft = el.as_dtype(dtype) if dtype is not None else _float_type(a.elt)
  if a.rank == 0:
    return cast(el.as_scalar(a), ft)
  axes = _axes(el, axis, a.rank)
  total = _reduce_prim(el, "add", a, axis, Const(0, ft), "mean", keepdims, ft)
  n = cast(_count(el, a, axes), ft)
  return el.apply_ufunc(UF["div"], [total, n])


def _var(el, a, axis, ddof, keepdims, sqrt: bool, dtype=None):
  a = _arr(el, a)
  ft = el.as_dtype(dtype) if dtype is not None else _float_type(a.elt)
  axes = _axes(el, axis, a.rank)
  if a.rank == 0:
    return Const(0.0, ft)
  a = el.cast_array(a, ft)
  if a.buf is None and a.cost > 4:
    a = el.as_concrete(a)
  m = mean(el, a, axis=axis, keepdims=True)
  m = el.as_concrete(m) if isinstance(m, ArrayVal) and m.buf is None else m
  dev = el.apply_ufunc(UF["sub"], [a, m])
  sq = el.apply_ufunc(UF["mul"], [dev, dev])
  total = _reduce_prim(el, "add", sq, axis, Const(0, ft), "var", keepdims)
  n = _count(el, a, axes)
  denom = cast(isub(n, el.as_index(ddof, "ddof")), ft)
  res = el.apply_ufunc(UF["div"], [total, denom])
  return el.apply_ufunc(UF["sqrt"], [res]) if sqrt else res


@register(np.var)
@method("var")
def var(el, a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  return _var(el, a, axis, ddof, keepdims, False, dtype)


@register(np.std)
@method("std")
def std(el, a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  return _var(el, a, axis, ddof, keepdims, True, dtype)


def _arg_extreme(el, a, axis, op: str, what: str, keepdims=False):
  a = _arr(el, a)
  if a.rank == 0:
    return Const(0, Int64)

  def lane(v: ArrayVal):
    v = _use(el, v)
    n = v.shape[0]
    el.check(cmp("gt", n, 0), "ValueError", f"attempt to get {what} of an empty sequence")

    def beats(x: Expr, best: Expr) -> Expr:
      better = ops.prim(op, (x, best), Bool)
      if v.elt.kind == "f":  # the first NaN wins, as in NumPy
        better = ops.land(ops.lor(better, ops.prim("isnan", (x,), Bool)), ops.lnot(ops.prim("isnan", (best,), Bool)))
      return better

    def fold(start: Expr, stop: Expr):
      """Best element of [start, stop) and where it is: a reduction whose
      accumulator carries a payload."""
      best = Var("best", v.elt)
      where = Var("where", Int64)
      el.mutables.update((best.uid, where.uid))
      el.emit(Assign(best, el.elem(v, (start,))))
      el.emit(Assign(where, cast(start, Int64)))

      def body(i):
        x = el.tmp(el.elem(v, (i,)), "x")
        el.emit(If(beats(x, best), [Assign(best, x), Assign(where, cast(i, Int64))], []))

      el.for_range(iadd(start, 1), stop, body)
      return best, where

    if not el.can_parallel((n,)):
      return el.copy_scalar(fold(ZERO, n)[1], "arg")
    # one (best, where) pair per chunk; chunks are in index order, so the
    # first of several equal elements still wins when they are combined
    nchunks = el.tmp(RtCall("num_chunks", (n, Const(True, Bool)), I), "nchunks")
    if isinstance(nchunks, Var):
      el.nonneg.add(nchunks.uid)
    bests = el.alloc_array(v.elt, (nchunks,), label="bests")
    wheres = el.alloc_array(Int64, (nchunks,), label="wheres")

    def chunk_body(chunk, start, stop):
      best, where = fold(start, stop)
      el.emit(Store(bests.buf, chunk, best))
      el.emit(Store(wheres.buf, chunk, where))

    el.par_loop(n, chunk_body, kind="reduce", nchunks=nchunks)
    best = Var("best", v.elt)
    where = Var("where", Int64)
    el.mutables.update((best.uid, where.uid))
    el.emit(Assign(best, Load(bests.buf, ZERO)))
    el.emit(Assign(where, Load(wheres.buf, ZERO)))

    def combine(c):
      x = el.tmp(Load(bests.buf, c), "x")
      el.emit(If(beats(x, best), [Assign(best, x), Assign(where, Load(wheres.buf, c))], []))

    el.for_range(ONE, nchunks, combine)
    return el.copy_scalar(where, "arg")

  if axis is None:
    return lane(el.flat_view(a) if a.rank != 1 else a)
  k = norm_axis(el, axis, a.rank)
  res = along_axis(el, a, k, lane, what)
  return _keepdims(el, res, a, [k], keepdims)


@register(np.argmin)
@method("argmin")
def argmin(el, a, axis=None, out=None, keepdims=False):
  return _arg_extreme(el, a, axis, "lt", "argmin", keepdims)


@register(np.argmax)
@method("argmax")
def argmax(el, a, axis=None, out=None, keepdims=False):
  return _arg_extreme(el, a, axis, "gt", "argmax", keepdims)


def _cumulative(el, a, axis, op: str, dtype):
  a = _arr(el, a)
  elt = el.as_dtype(dtype) if dtype is not None else _sum_type(a.elt)
  a = el.cast_array(a, elt)
  tape = el.tape
  if tape is not None and not tape.muted and op == "add" and elt.kind == "f" and tape.is_active(el, a):
    if a.buf is None and a.cost > CHEAP_COST:
      a = el.as_concrete(a)
    return tape.op(el, "cumsum", (a,), axis, lambda: cumulative_plain(el, a, axis, op))
  return cumulative_plain(el, a, axis, op)


def cumulative_plain(el, a: ArrayVal, axis, op: str):
  """Running fold of ``a`` with the primitive ``op`` along ``axis`` (None: flattened)."""
  from .adverbs import scan_axis
  from .core import scan_values

  spec = UF[op]

  def combine(x, y):
    return el.apply_ufunc(spec, [x, y])

  if a.rank == 0:
    a = el.expand_dims(a, [0])
  if axis is None or a.rank == 1:
    if a.rank != 1:
      a = el.flat_view(a)
    a = _use(el, a)
    return scan_values(el, a.shape[0], lambda i: el.tmp(el.elem(a, (i,)), "x"), combine, None, "cumulative")
  return scan_axis(el, a, norm_axis(el, axis, a.rank), combine)


@register(np.cumsum)
@method("cumsum")
def cumsum(el, a, axis=None, dtype=None, out=None):
  return _cumulative(el, a, axis, "add", dtype)


@register(np.cumprod)
@method("cumprod")
def cumprod(el, a, axis=None, dtype=None, out=None):
  return _cumulative(el, a, axis, "mul", dtype)


# ---------------------------------------------------------------------------
# element-wise helpers that are not ufuncs
# ---------------------------------------------------------------------------


def _sorted(el, a, axis, one_lane, what: str):
  """Apply a 1-d sorting routine along ``axis`` (every lane in parallel)."""
  a = _arr(el, a)
  if axis is None or a.rank <= 1:
    return one_lane(el, a)
  k = norm_axis(el, axis, a.rank)
  lanes = along_axis(el, a, k, lambda lane: one_lane(el, lane), what)
  # the sorted axis comes out last: put it back where it was
  perm = list(range(k)) + [a.rank - 1] + list(range(k, a.rank - 1))
  return el.transpose(el.as_concrete(lanes), perm)


@register(np.sort)
def sort(el, a, axis=-1, kind=None, order=None):
  """Sorted copy of ``a`` along ``axis`` (stable, whatever ``kind`` asks for; NaNs last)."""
  from . import pure

  tape = el.tape
  if tape is not None and not tape.muted and tape.is_active(el, _arr(el, a)):
    # differentiable: the sorted array is the input indexed by its sorting permutation
    a = _arr(el, a)
    if a.buf is None and a.cost > CHEAP_COST:
      a = el.as_concrete(a)
    if axis is None and a.rank > 1:
      a = el.flat_view(a)
    order = el.as_concrete(argsort(el, a, axis, kind))
    return take_along(el, a, order, -1 if axis is None else axis)
  return _sorted(el, a, axis, pure.sort1d.__parakeet_impl__, "sort")


def take_along(el, a: ArrayVal, index: ArrayVal, axis) -> ArrayVal:
  """``a`` with position ``index[...]`` picked along ``axis`` at every element."""
  k = norm_axis(el, axis, a.rank)
  el.consume(a)

  def at(idx):
    j = cast(el.tmp(el.elem(index, idx), "j"), I)
    return el.elem(a, tuple(idx[:k]) + (j,) + tuple(idx[k + 1 :]))

  return el.delayed(a.elt, a.shape, at, roots=el.union_roots(el.get_roots(a), el.get_roots(index)), cost=a.cost + 2)


@register(np.argsort)
@method("argsort")
def argsort(el, a, axis=-1, kind=None, order=None):
  """Indices that sort ``a`` along ``axis`` (stable, whatever ``kind`` asks for; NaNs last)."""
  from . import pure

  return _sorted(el, a, axis, pure.argsort1d.__parakeet_impl__, "argsort")


@register(np.where)
def where(el, condition, x=None, y=None):
  if x is None and y is None:
    return nonzero(el, condition)
  if x is None or y is None:
    raise el.error(TypeInferenceError, "either both or neither of x and y should be given")
  c, a, b = (el.operand(v, "where") for v in (condition, x, y))
  ta = a.elt if isinstance(a, ArrayVal) else a.type
  tb = b.elt if isinstance(b, ArrayVal) else b.type
  out = join(ta, tb)
  if not any(isinstance(v, ArrayVal) for v in (c, a, b)):
    return ops.select(el.truth(c), cast(a, out), cast(b, out))

  def f(cv, av, bv):
    return ops.select(cast(cv, Bool), cast(av, out), cast(bv, out))

  return el.elementwise([c, a, b], f, out.strong, cost=1)


@register(np.nonzero)
@method("nonzero")
def nonzero(el, a):
  a = _arr(el, a)
  if a.rank != 1:
    raise el.error(
            UnsupportedError, "nonzero() / one-argument where() is only supported for 1-d arrays",
            hint="use boolean-mask indexing (x[mask]) or np.where(cond, a, b)",
        )  # fmt: skip
  return (el.nonzero1d(el.cast_array(a, Bool)),)


@register(np.flatnonzero)
def flatnonzero(el, a):
  a = _arr(el, a)
  return el.nonzero1d(el.cast_array(el.flat_view(a) if a.rank != 1 else a, Bool))


@register(np.clip)
@method("clip")
def clip(el, a, a_min=None, a_max=None, out=None, min=None, max=None):  # noqa: A002
  lo = a_min if a_min is not None else min
  hi = a_max if a_max is not None else max
  res = el.operand(a, "clip")
  if lo is not None:
    res = el.apply_ufunc(UF["max"], [res, lo])
  if hi is not None:
    res = el.apply_ufunc(UF["min"], [res, hi])
  return res


@register(np.round, np.around)
@method("round")
def round_(el, a, decimals=0, out=None):
  a = el.operand(a, "round")
  d = el.static_int(decimals, "decimals") if not isinstance(decimals, int) else decimals
  t = a.elt if isinstance(a, ArrayVal) else a.type
  if t.kind != "f":
    return a
  if d == 0:
    return el.apply_ufunc(UF["rint"], [a])
  scale = Const(10.0**d, Float64 if t.weak else t)
  return el.apply_ufunc(UF["div"], [el.apply_ufunc(UF["rint"], [el.apply_ufunc(UF["mul"], [a, scale])]), scale])


@register(np.isclose)
def isclose(el, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  a, b = el.operand(a, "isclose"), el.operand(b, "isclose")
  if isinstance(a, ArrayVal) and a.buf is None and a.cost > 4:
    a = el.as_concrete(a)
  if isinstance(b, ArrayVal) and b.buf is None and b.cost > 4:
    b = el.as_concrete(b)
  diff = el.apply_ufunc(UF["abs"], [el.apply_ufunc(UF["sub"], [convert(el, Float64, a), convert(el, Float64, b)])])
  tol = el.apply_ufunc(
    UF["add"], [atol, el.apply_ufunc(UF["mul"], [rtol, el.apply_ufunc(UF["abs"], [convert(el, Float64, b)])])]
  )
  close = el.apply_ufunc(UF["le"], [diff, tol])
  same = el.apply_ufunc(UF["eq"], [a, b])  # also true for equal infinities
  res = el.apply_ufunc(UF["logical_or"], [close, same])
  if el.static_bool(equal_nan, "equal_nan") if not isinstance(equal_nan, bool) else equal_nan:
    both = el.apply_ufunc(UF["logical_and"], [el.apply_ufunc(UF["isnan"], [a]), el.apply_ufunc(UF["isnan"], [b])])
    res = el.apply_ufunc(UF["logical_or"], [res, both])
  return res


@register(np.allclose)
def allclose(el, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  res = isclose(el, a, b, rtol, atol, equal_nan)
  res = all_(el, res) if isinstance(res, ArrayVal) else res
  return cast(el.as_scalar(res), WeakBool)  # NumPy returns a Python bool here


@register(np.array_equal)
def array_equal(el, a1, a2):
  a, b = _arr(el, a1), _arr(el, a2)
  if a.rank != b.rank:
    return ops.FALSE
  if a.rank == 0:
    return cast(el.as_scalar(el.apply_ufunc(UF["eq"], [a, b])), WeakBool)
  same_shape: Expr = ops.TRUE
  for x, y in zip(a.shape, b.shape):
    if not el.dim_equal(x, y):
      same_shape = ops.land(same_shape, cmp("eq", x, y))
  if isinstance(same_shape, Const):
    if not same_shape.value:
      return ops.FALSE
    return cast(el.as_scalar(all_(el, el.apply_ufunc(UF["eq"], [a, b]))), WeakBool)
  same_shape = el.tmp(same_shape, "same")
  result = Var("equal", WeakBool)
  el.mutables.add(result.uid)
  el.emit(Assign(result, Const(False, WeakBool)))
  blk: list = []
  with el.in_block(blk):
    for x, y in zip(a.shape, b.shape):
      el.assume_equal(x, y) if isinstance(x, Var) and isinstance(y, Var) else None
    flat_a = el.broadcast_to(a, a.shape)
    mapping = [(k, ZERO, ONE) for k in range(b.rank)]
    b_as_a = el.remap(b, a.shape, mapping) if b.rank else b
    el.emit(Assign(result, cast(el.as_scalar(all_(el, el.apply_ufunc(UF["eq"], [flat_a, b_as_a]))), WeakBool)))
  el.emit(If(same_shape, blk, []))
  return result


@register(np.diff)
def diff(el, a, n=1, axis=-1):
  a = _arr(el, a)
  if a.rank == 0:
    raise el.error(TypeInferenceError, "diff requires input that is at least one dimensional")
  k = norm_axis(el, axis, a.rank)
  count = el.static_int(n, "n") if not isinstance(n, int) else n
  for _ in range(count):
    if a.buf is None and a.cost > 4:
      a = el.as_concrete(a)
    hi = el.getitem(a, tuple(slice(ONE, None) if d == k else slice(None) for d in range(a.rank)))
    lo = el.getitem(a, tuple(slice(None, Const(-1, I)) if d == k else slice(None) for d in range(a.rank)))
    a = el.apply_ufunc(UF["ne" if a.elt.kind == "b" else "sub"], [hi, lo])
  return a


@register(np.shape)
def shape_fn(el, a):
  return _arr(el, a).shape


@register(np.ndim)
def ndim_fn(el, a):
  return Const(_arr(el, a).rank, I)


@register(np.size)
def size_fn(el, a, axis=None):
  a = _arr(el, a)
  if axis is None:
    return _size(el, a)
  return a.shape[norm_axis(el, axis, a.rank)]


@register(np.real, np.conjugate, np.conj)
@method("conj", "conjugate")
def real(el, a):
  return a


@register(np.isscalar)
def isscalar(el, a):
  return Const(isinstance(a, Expr), Bool)


@register(np.dtype)
def dtype_fn(el, t):
  return TypeVal(el.as_dtype(t))


@register(np.result_type, np.promote_types)
def result_type(el, *args):
  t = None
  for a in args:
    if isinstance(a, TypeVal):
      cur = a.type
    elif isinstance(a, ArrayVal):
      cur = a.elt
    elif isinstance(a, Expr):
      cur = a.type
    elif isinstance(a, str):
      cur = el.as_dtype(a)
    else:
      raise el.error(TypeInferenceError, f"result_type() got {describe(a)}")
    t = cur if t is None else join(t, cur)
  if t is None:
    raise el.error(TypeInferenceError, "result_type() needs at least one argument")
  return TypeVal(t.strong)


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------


def _dot1(el, a: ArrayVal, b: ArrayVal):
  """Inner product of two 1-d arrays."""
  prod_ = el.apply_ufunc(UF["mul"], [a, b])
  elt = prod_.elt
  return _reduce_prim(el, "add", prod_, None, Const(0, elt), "dot")


def matmul(el, a, b, what: str = "matmul"):
  a, b = el.operand(a, "@"), el.operand(b, "@")
  if not isinstance(a, ArrayVal) or not isinstance(b, ArrayVal) or a.rank == 0 or b.rank == 0:
    if what == "dot":
      return el.apply_ufunc(UF["mul"], [a, b])
    raise el.error(TypeInferenceError, "matmul: input operands must be arrays, not scalars (use * instead)")
  if a.rank > 2 or b.rank > 2:
    return _stacked_matmul(el, a, b, what)
  a = el.as_concrete(a) if a.buf is None and a.cost > 2 else a
  b = el.as_concrete(b) if b.buf is None and b.cost > 2 else b
  inner_a, inner_b = a.shape[-1], b.shape[0]
  if not el.dim_equal(inner_a, inner_b):
    if isinstance(inner_a, Const) and isinstance(inner_b, Const):
      raise el.error(TypeInferenceError, f"{what}: shapes are not aligned ({inner_a.value} vs {inner_b.value})")
    el.check(
      cmp("eq", inner_a, inner_b), "ValueError", what + ": shapes are not aligned ({0} vs {1})", inner_a, inner_b
    )
    el.assume_equal(inner_a, inner_b)
  tape = el.tape
  if tape is not None and not tape.muted and (tape.is_active(el, a) or tape.is_active(el, b)):
    return tape.matmul(el, matmul_plain, a, b, what)
  return matmul_plain(el, a, b, what)


def _rows(el, a: ArrayVal, label: str = "rows") -> ArrayVal:
  """``a`` with rows that are contiguous in memory (copied if they are not):
  the inner loop of a product should walk both operands with stride one."""
  if a.buf is None:
    return a
  if isinstance(a.strides[-1], Const) and a.strides[-1].value == 1:
    return a
  return el.materialize(a, label)


def matmul_plain(el, a: ArrayVal, b: ArrayVal, what: str = "matmul"):
  """The product of two arrays of rank 1 or 2 whose inner dimensions agree."""
  if a.rank == 1 and b.rank == 1:
    return _dot1(el, a, b)
  if a.rank == 1:
    # a @ b is b.T @ a: the rows of b.T are what gets multiplied with a
    a, b = el.transpose(b), a
  a = _rows(el, a)
  el.consume(a)
  if b.rank == 1:
    el.consume(b)
    return build_map(el, (a.shape[0],), lambda idx: _dot1(el, el.axis_view(a, 0, idx[0], True), b), what)
  # the columns of b as rows: a view if b is itself a transpose, otherwise one copy
  bt = _rows(el, el.transpose(b), "bt")
  el.consume(bt)
  return build_map(
        el, (a.shape[0], bt.shape[0]),
        lambda idx: _dot1(el, el.axis_view(a, 0, idx[0], True), el.axis_view(bt, 0, idx[1], True)), what,
    )  # fmt: skip


def _stacked_matmul(el, a: ArrayVal, b: ArrayVal, what: str):
  """Products of stacks of matrices: the last two dimensions multiply, the
  ones before them broadcast (for ``dot``, they combine like an outer product)."""
  if what == "dot" and a.rank > 1 and b.rank > 2:
    raise el.error(UnsupportedError, "dot of two arrays with more than 2 dimensions is not supported; use @ or einsum")
  if b.rank <= 2:
    # every row of every matrix in the stack times b: one big matrix product
    lead = a.shape[:-1]
    flat = matmul(el, el.reshape(a, (Const(-1, I), a.shape[-1])), b, what)
    return el.reshape(el.as_array(flat), lead + (() if b.rank == 1 else (b.shape[1],)), checked=True)
  if a.rank == 1:
    a = el.expand_dims(a, [0])
    out = _stacked_matmul(el, a, b, what)
    return el.getitem(out, (Ellipsis, Const(0, I), slice(None)))
  # labels: batch dimensions right-aligned and shared, then i,k and k,j
  batch = max(a.rank, b.rank) - 2
  i, k, j = batch, batch + 1, batch + 2
  la = list(range(batch - (a.rank - 2), batch)) + [i, k]
  lb = list(range(batch - (b.rank - 2), batch)) + [k, j]
  return contract(el, [a, b], [la, lb], list(range(batch)) + [i, j], what, broadcast=True)


def contract(el, operands: list[ArrayVal], labels: list[list[int]], out: list[int], what: str, broadcast=False):
  """Einstein summation.  Each operand's dimensions carry labels; the result
  has the dimensions ``out``, and every label not in it is summed over.

  The operands are viewed in one index space (result labels, then summed
  ones), multiplied element by element, and the trailing block is reduced:
  a map of reductions, fused, parallel, tiled when it is two-dimensional."""
  summed = sorted({x for ls in labels for x in ls} - set(out))
  space = list(out) + summed
  sizes: dict[int, Expr] = {}
  views = []
  for arr, ls in zip(operands, labels):
    if len(set(ls)) != len(ls):
      raise el.error(UnsupportedError, f"{what}: a repeated subscript within one operand (a diagonal) is not supported")
    for x, d in zip(ls, arr.shape):
      known = sizes.get(x)
      if known is None or (broadcast and isinstance(known, Const) and known.value == 1):
        sizes[x] = d
      elif not el.dim_equal(known, d) and not (broadcast and isinstance(d, Const) and d.value == 1):
        if isinstance(known, Const) and isinstance(d, Const):
          raise el.error(TypeInferenceError, f"{what}: dimensions do not match ({known.value} vs {d.value})")
        el.check(cmp("eq", known, d), "ValueError", what + ": dimensions do not match ({0} vs {1})", known, d)
        el.assume_equal(known, d)
    if arr.buf is None and arr.cost > 2:
      arr = el.as_concrete(arr)
    order = sorted(range(len(ls)), key=lambda n: space.index(ls[n]))
    view = el.transpose(arr, order)
    missing = [n for n, x in enumerate(space) if x not in ls]
    views.append(el.expand_dims(view, missing) if missing else view)
  prod_ = views[0]
  for v in views[1:]:
    prod_ = el.apply_ufunc(UF["mul"], [prod_, v])
  if not summed:
    return prod_
  axes = tuple(Const(n, I) for n in range(len(out), len(space)))
  return _reduce_prim(el, "add", prod_, axes, Const(0, prod_.elt), what)


@register(np.einsum)
def einsum(el, subscripts, *operands):
  """``np.einsum`` for explicit or implicit subscripts without ellipses."""
  spec = el.require_static(subscripts, "the subscripts of einsum")
  if not isinstance(spec, str):
    raise el.error(TypeInferenceError, "einsum expects a subscript string such as 'ij,jk->ik' as its first argument")
  spec = spec.replace(" ", "")
  if "." in spec:
    raise el.error(UnsupportedError, "einsum with an ellipsis ('...') is not supported; name every dimension")
  lhs, arrow, rhs = spec.partition("->")
  terms = lhs.split(",")
  arrs = [_arr(el, x, "an operand of einsum") for x in operands]
  if len(terms) != len(arrs):
    raise el.error(TypeInferenceError, f"einsum: {len(terms)} subscript groups for {len(arrs)} operands")
  for t, arr in zip(terms, arrs):
    if not t.isalpha() and t != "":
      raise el.error(StaticValueError, f"einsum: invalid subscripts {t!r}")
    if len(t) != arr.rank:
      raise el.error(TypeInferenceError, f"einsum: subscripts {t!r} do not match an operand with {arr.rank} dimensions")
  letters = "".join(terms)
  if not arrow:
    rhs = "".join(sorted(c for c in set(letters) if letters.count(c) == 1))
  if len(set(rhs)) != len(rhs) or not set(rhs) <= set(letters):
    raise el.error(StaticValueError, f"einsum: invalid output subscripts {rhs!r}")
  if len(arrs) == 2 and len(terms[0]) == 2 and len(terms[1]) == 2 and len(rhs) == 2:
    (i, k), (k2, j) = terms
    if k == k2 and rhs == i + j and len({i, j, k}) == 3:
      return matmul(el, arrs[0], arrs[1], "einsum")  # exactly a matrix product
  return contract(el, arrs, [[ord(c) for c in t] for t in terms], [ord(c) for c in rhs], "einsum")


@register(np.matmul)
def _matmul(el, a, b):
  return matmul(el, a, b)


@register(np.dot)
@method("dot")
def dot(el, a, b):
  return matmul(el, a, b, "dot")


@register(np.vdot)
def vdot(el, a, b):
  a, b = _arr(el, a), _arr(el, b)
  return _dot1(el, el.flat_view(a) if a.rank != 1 else a, el.flat_view(b) if b.rank != 1 else b)


@register(np.inner)
def inner(el, a, b):
  a, b = el.operand(a, "inner"), el.operand(b, "inner")
  if isinstance(a, ArrayVal) and isinstance(b, ArrayVal) and a.rank == 1 and b.rank == 1:
    return matmul(el, a, b, "inner")
  if not isinstance(a, ArrayVal) or not isinstance(b, ArrayVal):
    return el.apply_ufunc(UF["mul"], [a, b])
  raise el.error(UnsupportedError, "inner() is only supported for 1-d arrays and scalars")


@register(np.outer)
def outer(el, a, b):
  a, b = _arr(el, a), _arr(el, b)
  a = el.flat_view(a) if a.rank != 1 else a
  b = el.flat_view(b) if b.rank != 1 else b
  return el.apply_ufunc(UF["mul"], [el.expand_dims(a, [1]), el.expand_dims(b, [0])])


@register(np.linalg.norm)
def norm(el, x, ord=None, axis=None, keepdims=False):  # noqa: A002
  x = _arr(el, x)
  ft = _float_type(x.elt)
  if x.rank == 0:
    return el.apply_ufunc(UF["abs"], [cast(el.as_scalar(x), ft)])
  o = None if ord is None else el.require_static(ord, "ord")
  if axis is None and x.rank > 2:
    raise el.error(UnsupportedError, "norm() of arrays with more than 2 dimensions needs an explicit axis")
  if axis is None and x.rank == 2 and o not in (None, "fro"):
    raise el.error(UnsupportedError, "only the Frobenius norm is supported for matrices; pass axis= for vector norms")
  xf = el.cast_array(x, ft)
  mag = el.apply_ufunc(UF["abs"], [xf])
  if o is None or o == 2 or o == "fro":
    total = _reduce_prim(el, "add", el.apply_ufunc(UF["mul"], [xf, xf]), axis, Const(0, ft), "norm", keepdims)
    return el.apply_ufunc(UF["sqrt"], [total])
  if o == 1:
    return _reduce_prim(el, "add", mag, axis, Const(0, ft), "norm", keepdims)
  if o == np.inf:
    return _reduce_prim(el, "max", mag, axis, None, "norm", keepdims)
  if o == -np.inf:
    return _reduce_prim(el, "min", mag, axis, None, "norm", keepdims)
  if o == 0:
    nz = el.cast_array(el.apply_ufunc(UF["ne"], [xf, Const(0, ft)]), ft)
    return _reduce_prim(el, "add", nz, axis, Const(0, ft), "norm", keepdims)
  if isinstance(o, (int, float)):
    p = Const(float(o), ft)
    total = _reduce_prim(el, "add", el.apply_ufunc(UF["pow"], [mag, p]), axis, Const(0, ft), "norm", keepdims)
    return el.apply_ufunc(UF["pow"], [total, Const(1.0 / o, ft)])
  raise el.error(StaticValueError, f"invalid norm order {o!r}")


_ = (Float32, Store, build_map)
