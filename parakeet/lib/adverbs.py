"""Parakeet's higher-order array operators: map, reduce, scan, allpairs, ...

Inside compiled code these are fused with their function arguments.  Called
directly from Python they compile a small wrapper on the fly, so
``parakeet.map(f, x)`` works without decorating anything.
"""

from __future__ import annotations

import builtins
import functools

from .. import prims
from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..syntax import ops
from ..syntax.ir import Const, Expr
from ..syntax.ops import I, cmp
from ..syntax.values import CHEAP_COST, ArrayVal, RangeVal, describe
from . import core
from .core import FUNCTIONS, NAMES, along_axis, build_map, norm_axis, reduce_arrays, reduce_values, scan_values


class LibFunction:
  """A library function usable inside compiled code *and* callable from Python."""

  def __init__(self, name: str, impl, doc: str | None = None):
    self.__name__ = name
    self.__qualname__ = name
    self.__doc__ = doc or impl.__doc__
    self.__module__ = "parakeet"
    self.impl = impl
    FUNCTIONS[self] = impl
    NAMES[self] = name

  def __repr__(self) -> str:
    return f"<parakeet function {self.__name__}>"

  def __call__(self, *args, **kwargs):
    from ..api import call_staged

    return call_staged(self, args, kwargs)


def libfunction(name: str | None = None):
  def deco(impl):
    return LibFunction(name or impl.__name__, impl)

  return deco


def _arrayish(el, v):
  """Arrays stay arrays; tuples/ranges become arrays; scalars pass through."""
  if isinstance(v, (tuple, RangeVal)):
    return el.as_array(v)
  return v


def _prepare(el, a: ArrayVal) -> ArrayVal:
  """Make ``a`` safe to slice once per iteration of a data-parallel loop."""
  if a.buf is None and a.cost > CHEAP_COST:
    a = el.as_concrete(a)
  el.consume(a)
  return a


def _axis_arg(el, axis, what: str = "axis"):
  if axis is None:
    return None
  if isinstance(axis, tuple):
    return tuple(el.static_int(a, what) for a in axis)
  return el.static_int(axis, what)


@libfunction("map")
def map(el, f, *args, axis=0):  # noqa: A001
  """Apply ``f`` to every element along ``axis`` of the argument arrays.

  With ``axis=None`` the function is applied to every scalar element (with
  broadcasting).  Non-array arguments are passed to every call unchanged."""
  if not args:
    raise el.error(TypeInferenceError, "map() needs at least one array argument")
  vals = [_arrayish(el, a) for a in args]
  arrays = [a for a in vals if isinstance(a, ArrayVal) and a.rank > 0]
  if not arrays:
    return el.call(f, [el.as_scalar(v) if isinstance(v, ArrayVal) else v for v in vals])
  ax = _axis_arg(el, axis)
  if ax is None:
    shape = el.broadcast_shapes([a.shape for a in arrays], "arguments of map")
    adapted = []
    for v in vals:
      if isinstance(v, ArrayVal):
        if v.rank == 0:
          adapted.append(el.as_scalar(v))
        else:
          v = _prepare(el, v)
          adapted.append(el.broadcast_to(v, shape, consumed=True))
      else:
        adapted.append(v)

    def at(idx):
      return el.call(f, [el.tmp(el.elem(a, idx), "x") if isinstance(a, ArrayVal) else a for a in adapted])

    return build_map(el, shape, at, "map")
  if isinstance(ax, tuple):
    raise el.error(StaticValueError, "map() takes a single axis (or None)")
  n = None
  prepared = []
  for v in vals:
    if isinstance(v, ArrayVal) and v.rank > 0:
      k = norm_axis(el, ax, v.rank)
      v = _prepare(el, v)
      dim = v.shape[k]
      if n is None:
        n = dim
      elif not el.dim_equal(n, dim):
        el.check(cmp("eq", n, dim), "ValueError", "arguments of map have different lengths ({0} and {1})", n, dim)
      prepared.append((v, k))
    elif isinstance(v, ArrayVal):
      prepared.append((el.as_scalar(v), None))
    else:
      prepared.append((v, None))

  def at(idx):
    return el.call(f, [v if k is None else el.axis_view(v, k, idx[0], consumed=True) for v, k in prepared])

  return build_map(el, (n,), at, "map")


each = map


@libfunction("allpairs")
def allpairs(el, f, x, y, axis=0):
  """``result[i, j] = f(x[i], y[j])`` for all pairs of elements along ``axis``."""
  return outer_map.impl(el, f, x, y, axis=axis)


@libfunction("outer_map")
def outer_map(el, f, *args, axis=0, axes=None):
  """Apply ``f`` to every combination of elements of the argument arrays."""
  if axes is not None:
    axis = axes
  ax = _axis_arg(el, axis)
  vals = [el.as_array(_arrayish(el, a)) for a in args]
  if not vals:
    raise el.error(TypeInferenceError, "outer_map() needs at least one array argument")
  if not isinstance(ax, tuple):
    ax = (ax,) * len(vals)
  if len(ax) != len(vals):
    raise el.error(StaticValueError, f"outer_map() got {len(vals)} arrays but {len(ax)} axes")
  prepared = []
  shape = []
  for v, a in zip(vals, ax):
    if v.rank == 0:
      raise el.error(TypeInferenceError, "outer_map() arguments must be at least 1-dimensional")
    if a is None:
      v = _prepare(el, el.flat_view(v))
      k = 0
    else:
      k = norm_axis(el, a, v.rank)
      v = _prepare(el, v)
    prepared.append((v, k))
    shape.append(v.shape[k])

  def at(idx):
    return el.call(f, [el.axis_view(v, k, i, consumed=True) for (v, k), i in zip(prepared, idx)])

  return build_map(el, tuple(shape), at, "allpairs")


def _reduce(el, f, x, init, axis, what: str):
  x = _arrayish(el, x)
  if not isinstance(x, ArrayVal):
    raise el.error(TypeInferenceError, f"{what}() expects an array, got {describe(x)}")
  if isinstance(init, ArrayVal) and init.rank == 0:
    init = el.as_scalar(init)

  def combine(a, b):
    return el.call(f, [a, b])

  ax = _axis_arg(el, axis)
  if isinstance(ax, tuple):
    raise el.error(StaticValueError, f"{what}() takes a single axis (or None)")
  if ax is None and not isinstance(init, ArrayVal):
    if x.rank == 0:
      v = el.as_scalar(x)
      return v if init is None else combine(init, v)
    x = _prepare(el, x)
    return reduce_values(el, x.shape, lambda idx: el.tmp(el.elem(x, idx), "x"), combine, init, what)
  k = norm_axis(el, 0 if ax is None else ax, x.rank)
  if x.rank == 1 and not isinstance(init, ArrayVal):
    x = _prepare(el, x)
    return reduce_values(el, x.shape, lambda idx: el.tmp(el.elem(x, idx), "x"), combine, init, what)
  if isinstance(f, prims.UFunc) and (init is None or isinstance(init, Expr)):
    # element-wise combiner: reduce each 1-d lane independently (and in parallel)
    return along_axis(
            el, x, k,
            lambda lane: reduce_values(
                el, lane.shape, lambda idx: el.tmp(el.elem(lane, idx), "x"), combine, init, what
            ),
            what,
        )  # fmt: skip
  x = _prepare(el, x)
  return reduce_arrays(el, x.shape[k], lambda i: el.axis_view(x, k, i, consumed=True), combine, init, what)


@libfunction("reduce")
def reduce(el, f, x, init=None, axis=None):  # noqa: A001
  """Fold ``f`` over the elements of ``x`` (all of them, or slices along ``axis``).

  ``f`` should be associative: recognised primitives (add, multiply, min,
  max, logical/bitwise and/or) are reduced in parallel."""
  return _reduce(el, f, x, init, axis, "reduce")


@core.register(functools.reduce)
def _functools_reduce(el, f, x, initial=None):
  x = _arrayish(el, x)
  if isinstance(x, ArrayVal):
    return _reduce(el, f, x, initial, 0, "reduce")
  raise el.error(TypeInferenceError, f"reduce() expects an array or tuple, got {describe(x)}")


@libfunction("scan")
def scan(el, f, x, init=None, axis=0):
  """Running fold: ``result[i] = f(result[i - 1], x[i])``."""
  x = el.as_array(_arrayish(el, x))
  ax = _axis_arg(el, axis)
  if isinstance(init, ArrayVal) and init.rank == 0:
    init = el.as_scalar(init)
  if init is not None and not isinstance(init, Expr):
    raise el.error(UnsupportedError, "scan() with an array-valued 'init' is not supported")

  def combine(a, b):
    return el.call(f, [a, b])

  if ax is None or x.rank == 1:
    if x.rank != 1:
      x = el.flat_view(x)
    x = _prepare(el, x)
    return scan_values(el, x.shape[0], lambda i: el.tmp(el.elem(x, (i,)), "x"), combine, init)
  return scan_axis(el, x, norm_axis(el, ax, x.rank), combine, init)


def scan_axis(el, x: ArrayVal, k: int, combine, init=None) -> ArrayVal:
  """Running fold along axis ``k`` of a multi-dimensional array."""
  from ..ndtypes import join
  from ..syntax.ir import Assign, Store, Var

  x = _prepare(el, x)
  # accumulator type
  acc_type = init.type if init is not None else x.elt
  el.loop_depth += 1
  try:
    for _ in range(6):
      with el.trial():
        r = combine(Var("acc", acc_type), Var("x", x.elt))
        if not isinstance(r, Expr):
          raise el.error(TypeInferenceError, "the function passed to scan must return a scalar")
        r_type = r.type
      new = join(acc_type, r_type)
      if new is acc_type:
        break
      acc_type = new
  finally:
    el.loop_depth -= 1
  out = el.alloc_array(acc_type.strong, x.shape, label="scan")
  lead = x.shape[:k] + x.shape[k + 1 :]
  n = x.shape[k]

  def lane(idx):
    full = lambda i: tuple(idx[:k]) + (i,) + tuple(idx[k:])  # noqa: E731
    acc = Var("acc", acc_type)
    el.mutables.add(acc.uid)
    if init is not None:
      el.emit(Assign(acc, ops.cast(init, acc_type)))

      def body(i):
        el.emit(Assign(acc, ops.cast(el.as_scalar(combine(acc, el.tmp(el.elem(x, full(i)), "x"))), acc_type)))
        el.emit(Store(out.buf, el.index_expr(out, full(i)), ops.cast(acc, acc_type.strong)))

      el.for_range(ops.ZERO, n, body)
    else:
      from ..syntax.ir import If

      first: list = []
      with el.in_block(first):
        el.emit(Assign(acc, ops.cast(el.elem(x, full(ops.ZERO)), acc_type)))
        el.emit(Store(out.buf, el.index_expr(out, full(ops.ZERO)), ops.cast(acc, acc_type.strong)))
      el.emit(If(cmp("gt", n, 0), first, []))

      def body(i):
        el.emit(Assign(acc, ops.cast(el.as_scalar(combine(acc, el.tmp(el.elem(x, full(i)), "x"))), acc_type)))
        el.emit(Store(out.buf, el.index_expr(out, full(i)), ops.cast(acc, acc_type.strong)))

      el.for_range(ops.ONE, n, body)

  el.loop_nest(lead, lane)
  return out


def _index_value(idx, scalar_index: bool):
  return idx[0] if scalar_index else tuple(idx)


@libfunction("imap")
def imap(el, f, shape):
  """Array of the given ``shape`` with ``result[idx] = f(idx)``.

  ``f`` receives an integer when ``shape`` is an integer, otherwise a tuple."""
  scalar_index = isinstance(shape, Expr) or (isinstance(shape, ArrayVal) and shape.rank == 0)
  dims = el.check_dims(el.shape_tuple(shape, "the shape passed to imap"))
  return build_map(el, dims, lambda idx: el.call(f, [_index_value(idx, scalar_index)]), "imap")


@libfunction("ireduce")
def ireduce(el, f, shape, init=None, combine=None):
  """Reduce ``f(idx)`` over every index of ``shape`` with ``combine``
  (defaults to addition)."""
  scalar_index = isinstance(shape, Expr)
  dims = el.check_dims(el.shape_tuple(shape, "the shape passed to ireduce"))
  comb = combine if combine is not None else prims.UFUNCS["add"]
  return reduce_values(
        el, dims, lambda idx: el.as_scalar(el.call(f, [_index_value(idx, scalar_index)]), "ireduce element"),
        lambda a, b: el.call(comb, [a, b]), init, "ireduce",
    )  # fmt: skip


@libfunction("parfor")
def parfor(el, f, bounds):
  """Call ``f(idx)`` for every index below ``bounds``, in parallel, for its
  side effects (typically writes to a preallocated output array)."""
  scalar_index = isinstance(bounds, Expr)
  dims = el.check_dims(el.shape_tuple(bounds, "the bounds passed to parfor"))

  def body(idx):
    el.call(f, [_index_value(idx, scalar_index)])

  el.loop_nest(dims, body)
  return None


@libfunction("filter")
def filter(el, pred, x):  # noqa: A001
  """Elements of ``x`` (along its first axis) for which ``pred`` is true."""
  x = el.as_array(_arrayish(el, x))
  mask = map.impl(el, pred, x, axis=0)
  if not isinstance(mask, ArrayVal) or mask.rank != 1:
    raise el.error(TypeInferenceError, "the predicate passed to filter must return a scalar")
  mask = el.cast_array(mask, _bool())
  return el.getitem(el.as_concrete(x) if x.buf is None and x.cost > CHEAP_COST else x, mask)


def _bool():
  from ..ndtypes import Bool

  return Bool


# ---------------------------------------------------------------------------
# grad: the adverb that turns a function into its derivative
# ---------------------------------------------------------------------------


@libfunction("gradient")
def _gradient(el, f, *args, argnum=0, **kwargs):
  from ..transforms import autodiff

  return autodiff.differentiate(el, f, argnum, list(args), kwargs, False, "grad")


@libfunction("value_and_gradient")
def _value_and_gradient(el, f, *args, argnum=0, **kwargs):
  from ..transforms import autodiff

  return autodiff.differentiate(el, f, argnum, list(args), kwargs, True, "value_and_grad")


class Gradient:
  """What :func:`grad` returns: call it like the function it differentiates."""

  def __init__(self, fn, argnum, impl: LibFunction):
    if not callable(fn):
      raise TypeError(f"{impl.__name__.replace('ient', '')} expects a function, got {type(fn).__name__}")
    nums = argnum if isinstance(argnum, tuple) else (argnum,)
    if not nums or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in nums):
      raise TypeError("argnum must be a non-negative integer or a tuple of them")
    self.fn = fn
    self.argnum = argnum
    self.impl = impl
    self.__name__ = getattr(fn, "__name__", "function")
    self.__doc__ = getattr(fn, "__doc__", None)

  def __repr__(self) -> str:
    return f"<parakeet {self.impl.__name__} of {self.__name__}>"

  def __call__(self, *args, **kwargs):
    from ..api import call_staged

    return call_staged(self.impl, (self.fn, *args), {**kwargs, "argnum": self.argnum})

  def __parakeet_value__(self, el):
    from ..syntax.values import Partial

    return Partial(self.impl, (el.lift(self.fn),), {"argnum": el.lift(self.argnum)})


def grad(f, argnum=0):
  """The gradient of the scalar-valued function ``f`` with respect to its
  argument number ``argnum`` (or, for a tuple, to each of those arguments).

  ``grad(f)(*args)`` takes the arguments of ``f`` and returns an array shaped
  like the differentiated argument (a tuple of arrays for a tuple of
  argument numbers; nested tuples of parameters give nested tuples of
  gradients).  Works inside compiled code and from plain Python."""
  return Gradient(f, argnum, _gradient)


def value_and_grad(f, argnum=0):
  """Like :func:`grad`, but the new function returns ``(f(*args), gradient)``."""
  return Gradient(f, argnum, _value_and_gradient)


@core.register(grad)
def _grad_staged(el, f, argnum=0):
  from ..syntax.values import Partial

  return Partial(_gradient, (f,), {"argnum": argnum})


@core.register(value_and_grad)
def _value_and_grad_staged(el, f, argnum=0):
  from ..syntax.values import Partial

  return Partial(_value_and_gradient, (f,), {"argnum": argnum})


@core.register(builtins.map)
def _builtin_map(el, f, *args):
  return map.impl(el, f, *args, axis=0)


@core.register(builtins.filter)
def _builtin_filter(el, pred, x):
  return filter.impl(el, pred, x)


__all__ = [
    "LibFunction", "map", "each", "allpairs", "outer_map", "reduce", "scan", "imap", "ireduce",
    "parfor", "filter", "scan_axis", "grad", "value_and_grad", "Gradient",
]  # fmt: skip
_ = (Const, I)
