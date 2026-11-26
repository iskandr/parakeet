"""Staged implementations of Python builtins and the ``math`` module."""

from __future__ import annotations

import builtins
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import types as pytypes

import numpy as np

from .. import prims
from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..index import FrozenHashTable, SortedIndex
from ..ndtypes import Bool, WeakBool, WeakFloat, WeakInt
from ..ragged import RaggedArray
from ..syntax import ops
from ..syntax.ir import Const, Expr
from ..syntax.ops import I, cast, cmp
from ..syntax.records import ComplexArray, ComplexScalar, ComplexTypeVal, StructuredView, record_type
from ..syntax.values import (
  ArrayVal,
  BatchedVal,
  ChainVal,
  EnumerateVal,
  FixedListVal,
  ListVal,
  ProductVal,
  RangeVal,
  TypeVal,
  VecVal,
  ZipVal,
  describe,
)
from . import numpy_lib as npl
from .core import method, norm_axis, reduce_values, register

UF = prims.UFUNCS


@register(len)
def _len(el, x):
  if isinstance(x, ArrayVal):
    if x.rank == 0:
      raise el.error(TypeInferenceError, "len() of a 0-d array")
    return x.shape[0]
  if isinstance(x, ComplexArray):
    if x.rank == 0:
      raise el.error(TypeInferenceError, "len() of a 0-d complex array")
    return x.shape[0]
  if isinstance(x, StructuredView):
    if x.rank == 0:
      raise el.error(TypeInferenceError, "len() of a 0-d structured array")
    return x.shape[0]
  if isinstance(x, RaggedArray):
    if el.options.bounds_check:
      el.check(cmp("ge", x.offsets.shape[0], ops.ONE), "ValueError", "RaggedArray offsets must have length >= 1")
    return ops.isub(x.offsets.shape[0], ops.ONE)
  if isinstance(x, (SortedIndex, FrozenHashTable)):
    return x.keys.shape[0]
  if isinstance(x, ListVal):
    return Const(len(x.items), I)
  if isinstance(x, (VecVal, FixedListVal)):
    return x.length
  if isinstance(x, (tuple, str)):
    return Const(len(x), I)
  if isinstance(x, RangeVal):
    return el.range_length(x)
  if isinstance(x, (ZipVal, EnumerateVal, ChainVal, ProductVal, BatchedVal)):
    return el.iter_source(x)[0]
  raise el.error(TypeInferenceError, f"object of type {describe(x)} has no len()")


@register(ord)
def _ord(el, c):
  if isinstance(c, str):
    if len(c) != 1:
      raise el.error(TypeInferenceError, f"ord() expected a character, but string of length {len(c)} found")
    return Const(ord(c), WeakInt)
  if isinstance(c, ArrayVal) and c.elt.name == "uint8" and c.rank == 1 and isinstance(c.shape[0], Const):
    if c.shape[0].value != 1:
      raise el.error(TypeInferenceError, f"ord() expected a byte of length 1, got length {c.shape[0].value}")
    return cast(el.elem(c, (ops.ZERO,)), WeakInt)
  raise el.error(TypeInferenceError, f"ord() expected a character literal, got {describe(c)}")


@register(dataclasses.replace)
def _dataclass_replace(el, obj, **changes):
  rec = record_type(obj)
  if rec is None or not dataclasses.is_dataclass(rec.cls):
    raise el.error(TypeInferenceError, f"replace() expects a dataclass instance, got {describe(obj)}")
  if hasattr(rec.cls, "__post_init__"):
    raise el.error(UnsupportedError, f"replace() on dataclass {rec.cls.__name__} with __post_init__ is not supported")
  init_fields = {f.name for f in dataclasses.fields(rec.cls) if f.init}
  non_init = set(changes) & (set(rec.names) - init_fields)
  if non_init:
    raise el.error(
      TypeInferenceError,
      f"field(s) {', '.join(sorted(non_init))} of {rec.cls.__name__} are declared with init=False",
    )
  unknown = set(changes) - set(rec.names)
  if unknown:
    raise el.error(TypeInferenceError, f"{rec.cls.__name__} has no field(s): {', '.join(sorted(unknown))}")
  vals = [changes[name] if name in changes else getattr(obj, name) for name in rec.names]
  return rec.build(vals)


@method("_replace")
def _namedtuple_replace(el, obj, **changes):
  rec = record_type(obj)
  if rec is None or not issubclass(rec.cls, tuple):
    raise el.error(TypeInferenceError, f"_replace() expects a NamedTuple instance, got {describe(obj)}")
  unknown = set(changes) - set(rec.names)
  if unknown:
    raise el.error(TypeInferenceError, f"got unexpected field name(s): {', '.join(sorted(unknown))}")
  vals = [changes[name] if name in changes else getattr(obj, name) for name in rec.names]
  return rec.build(vals)


@register(RaggedArray.from_sequences, name="RaggedArray.from_sequences")
def _ragged_from_sequences(el, sequences, dtype=None):
  from ..ndtypes import Int64
  from ..syntax.ir import Store

  if isinstance(sequences, ListVal):
    items = list(sequences.items)
  elif isinstance(sequences, tuple):
    items = list(sequences)
  else:
    raise el.error(
      TypeInferenceError,
      f"RaggedArray.from_sequences() expects a static list or tuple of arrays, got {describe(sequences)}",
    )
  n_segs = len(items)
  if n_segs == 0:
    elt = el.as_dtype(dtype) if dtype is not None else el.as_dtype(np.float32)
    vals = el.alloc_array(elt, (ops.ZERO,), zero=False, label="rag_vals")
    offs = el.alloc_array(Int64, (ops.ONE,), zero=True, label="rag_offs")
    return RaggedArray(vals, offs)
  arrs = [el.as_array(x, "RaggedArray sequence") for x in items]
  elt = el.as_dtype(dtype) if dtype is not None else arrs[0].elt
  if dtype is None:
    for a in arrs[1:]:
      from ..ndtypes import join as _join

      elt = _join(elt, a.elt)
  values = npl.concatenate(el, tuple(arrs), axis=0, dtype=TypeVal(elt))
  offs = el.alloc_array(Int64, (Const(n_segs + 1, I),), zero=False, label="rag_offs")
  running: Expr = ops.ZERO
  el.emit(Store(offs.buf, ops.ZERO, ops.ZERO))
  for k, a in builtins.enumerate(arrs):
    running = el.tmp(ops.iadd(running, a.shape[0]), "roff")
    el.emit(Store(offs.buf, Const(k + 1, I), running))
  el.record_store(offs)
  return RaggedArray(values, offs)


@method("append")
def _list_append(el, obj, value):
  if isinstance(obj, ListVal):
    return el.list_append(obj, value)
  if isinstance(obj, VecVal):
    return el.vec_append(obj, value)
  if isinstance(obj, FixedListVal):
    return el.fixed_list_append(obj, value)
  raise el.error(TypeInferenceError, f"{describe(obj)} has no method 'append'")


@method("extend")
def _list_extend(el, obj, values):
  if isinstance(obj, ListVal):
    return el.list_extend(obj, values)
  if isinstance(obj, VecVal):
    return el.vec_extend(obj, values)
  if isinstance(obj, FixedListVal):
    return el.fixed_list_extend(obj, values)
  raise el.error(TypeInferenceError, f"{describe(obj)} has no method 'extend'")


@method("pop")
def _list_pop(el, obj, index=None):
  if isinstance(obj, ListVal):
    return el.list_pop(obj, index)
  if isinstance(obj, VecVal):
    if index is not None:
      raise el.error(UnsupportedError, "pop(index) on runtime list is not supported; use pop()")
    return el.vec_pop(obj)
  if isinstance(obj, FixedListVal):
    if index is not None:
      raise el.error(UnsupportedError, "pop(index) on fixed_list is not supported; use pop()")
    return el.fixed_list_pop(obj)
  raise el.error(TypeInferenceError, f"{describe(obj)} has no method 'pop'")


@method("clear")
def _list_clear(el, obj):
  if isinstance(obj, ListVal):
    return el.list_clear(obj)
  if isinstance(obj, VecVal):
    return el.vec_clear(obj)
  if isinstance(obj, FixedListVal):
    return el.fixed_list_clear(obj)
  raise el.error(TypeInferenceError, f"{describe(obj)} has no method 'clear'")


@method("bit_count")
def _bit_count(el, x):
  if not isinstance(x, Expr) or x.type.kind not in "iub":
    raise el.error(TypeInferenceError, f"bit_count() requires an integer scalar, got {describe(x)}")
  if x.type.kind == "i":
    from ..ndtypes import ScalarType

    u_t = ScalarType(f"uint{x.type.bits}")
    u = cast(x, u_t)
    neg_u = ops.prim("sub", (Const(0, u_t), u), u_t)
    v = ops.select(cmp("lt", x, Const(0, x.type)), neg_u, u)
  else:
    v = x
  return el.apply_ufunc(UF["popcount"], [v], python=x.type.weak)


@register(range)
def _range(el, start, stop=None, step=None):
  if stop is None:
    start, stop = Const(0, I), start
  if step is None:
    step = Const(1, I)
  parts = []
  for v, what in ((start, "start"), (stop, "stop"), (step, "step")):
    v = el.as_scalar(v, f"range() {what}")
    if v.type.kind not in "iub":
      raise el.error(TypeInferenceError, f"range() arguments must be integers, got {describe(v)}")
    parts.append(el.tmp(cast(v, I), what))
  return RangeVal(*parts)


@register(zip)
def _zip(el, *items, strict=False):
  if not items:
    return ()
  flag = el.truth(el.lift(strict) if type(strict) is bool else strict)
  if all(isinstance(x, tuple) for x in items) and (
    len({len(x) for x in items}) == 1 or (isinstance(flag, Const) and not flag.value)
  ):
    return tuple(builtins.zip(*items))
  return ZipVal(tuple(items), flag)


@register(enumerate)
def _enumerate(el, x, start=0):
  s = Const(start, I) if isinstance(start, int) else el.as_index(start, "start")
  if isinstance(x, tuple):
    if not isinstance(s, Const):
      raise el.error(StaticValueError, "enumerate() over a tuple needs a constant start")
    return tuple((Const(s.value + k, I), v) for k, v in builtins.enumerate(x))
  return EnumerateVal(x, s)


@register(reversed)
def _reversed(el, x):
  if isinstance(x, tuple):
    return tuple(builtins.reversed(x))
  if isinstance(x, RangeVal):
    n = el.range_length(x)
    last = el.tmp(ops.iadd(x.start, ops.imul(ops.isub(n, 1), x.step)), "last")
    return RangeVal(last, el.tmp(ops.isub(x.start, x.step)), el.tmp(ops.isub(0, x.step)))
  if isinstance(x, ArrayVal):
    return npl.flip(el, x, 0)
  raise el.error(TypeInferenceError, f"{describe(x)} is not reversible")


@register(tuple)
def _tuple(el, x=()):
  if isinstance(x, ListVal):
    return tuple(x.items)
  if isinstance(x, tuple):
    return x
  if isinstance(x, ArrayVal) and x.rank >= 1 and isinstance(x.shape[0], Const):
    return tuple(el.getitem(x, Const(k, I)) for k in builtins.range(x.shape[0].value))
  raise el.error(
    StaticValueError,
    f"tuple() needs a tuple or an array whose length is known at compile time, got {describe(x)}",
  )


@register(list)
def _list(el, x=()):
  if isinstance(x, ListVal):
    return ListVal(list(x.items), origin=x.origin, path=x.path, loop_depth=el.loop_depth, branch_depth=len(el.if_stack))
  if isinstance(x, VecVal):
    return el.vec_to_array(x)
  if isinstance(x, FixedListVal):
    return el.fixed_list_to_array(x)
  if isinstance(x, (ChainVal, ProductVal, BatchedVal, ZipVal, EnumerateVal)):
    return el.as_array(x)
  if isinstance(x, tuple) and not builtins.all(isinstance(v, Expr) for v in x):
    ranks = {v.rank if isinstance(v, ArrayVal) else None for v in x}
    if None in ranks or len(ranks) != 1:
      return ListVal(list(x), loop_depth=el.loop_depth, branch_depth=len(el.if_stack))
  return el.as_array(x, "argument of list()")


@register(slice)
def _slice(el, start, stop=None, step=None):
  if stop is None and step is None:
    return slice(None, start, None)
  return slice(start, stop, step)


@register(abs)
def _abs(el, x):
  return el.apply_ufunc(UF["abs"], [x], python=True)


@register(pow)
def _pow(el, x, y):
  return el.binop("pow", x, y)


@register(divmod)
def _divmod(el, a, b):
  a, b = el.bindable(a, "a"), el.bindable(b, "b")
  return (el.binop("floordiv", a, b), el.binop("mod", a, b))


@register(round)
def _round(el, x, ndigits=None):
  if isinstance(x, ArrayVal):
    return npl.round_(el, x, 0 if ndigits is None else ndigits)
  x = el.as_scalar(x, "argument of round()")
  if x.type.kind != "f":
    return x
  if ndigits is None:
    r = ops.prim("rint", (x,), x.type)
    return cast(r, WeakInt) if x.type.weak else r
  res = npl.round_(el, x, ndigits)
  return cast(res, WeakFloat) if x.type.weak else res


def _fold(el, name: str, values, python: bool = True):
  acc = values[0]
  for v in values[1:]:
    acc = el.apply_ufunc(UF[name], [acc, v], python=python)
  return acc


def _minmax(el, name: str, args, key, default):
  if key is not None or default is not None:
    raise el.error(UnsupportedError, f"the 'key' and 'default' arguments of {name}() are not supported")
  if len(args) == 0:
    raise el.error(TypeInferenceError, f"{name} expected at least 1 argument, got 0")
  if len(args) > 1:
    return _fold(el, name, list(args))
  (x,) = args
  if isinstance(x, tuple):
    if not x:
      raise el.error(StaticValueError, f"{name}() arg is an empty sequence")
    return _fold(el, name, list(x))
  if isinstance(x, RangeVal):
    x = el.range_array(x)
  if isinstance(x, ArrayVal):
    return npl.amin(el, x) if name == "min" else npl.amax(el, x)
  raise el.error(TypeInferenceError, f"{describe(x)} is not iterable")


@register(min)
def _min(el, *args, key=None, default=None):
  return _minmax(el, "min", args, key, default)


@register(max)
def _max(el, *args, key=None, default=None):
  return _minmax(el, "max", args, key, default)


@register(sum)
def _sum(el, x, start=None):
  if start is None:
    start = Const(0, WeakInt)
  if isinstance(x, ListVal):
    x = tuple(x.items)
  elif isinstance(x, VecVal):
    x = el.vec_to_array(x)
  elif isinstance(x, FixedListVal):
    x = el.fixed_list_to_array(x)
  elif isinstance(x, ChainVal):
    acc = start
    for part in x.items:
      acc = _sum(el, part, acc)
    return acc
  if isinstance(x, tuple):
    acc = start
    for v in x:
      acc = el.binop("add", acc, v)
    return acc
  add = UF["add"]

  def combine(a, b):
    return el.apply_ufunc(add, [a, b], python=True)

  if isinstance(x, RangeVal) and isinstance(start, Expr):
    # the elements of a range are Python ints
    n = el.range_length(x)
    first, step = el.tmp(x.start), el.tmp(x.step)
    return reduce_values(el, (n,), lambda idx: ops.iadd(first, ops.imul(idx[0], step)), combine, start, "sum")
  if isinstance(x, RangeVal):
    x = el.range_array(x)
  if not isinstance(x, ArrayVal):
    raise el.error(TypeInferenceError, f"{describe(x)} is not iterable")
  if x.rank == 0:
    raise el.error(TypeInferenceError, "iteration over a 0-d array")

  def lane(v):
    v = npl._use(el, v)
    init = start if isinstance(start, Expr) else None
    return reduce_values(el, v.shape, lambda idx: el.tmp(el.elem(v, idx), "x"), combine, init, "sum")

  if x.rank == 1:
    if isinstance(start, ArrayVal):
      return el.binop("add", start, lane(x))
    return lane(x)
  # Python's sum over an n-d array adds its sub-arrays: reduce along axis 0
  from .core import along_axis

  res = along_axis(el, x, 0, lane, "sum")
  if isinstance(start, ArrayVal):
    return el.binop("add", start, res)
  return res


def _anyall(el, x, name: str):
  if isinstance(x, tuple):
    acc: Expr = ops.FALSE if name == "any" else ops.TRUE
    for v in x:
      t = el.truth(v)
      acc = ops.lor(acc, t) if name == "any" else ops.land(acc, t)
    return acc
  if isinstance(x, RangeVal):
    x = el.range_array(x)
  if isinstance(x, ArrayVal):
    res = npl.any_(el, x) if name == "any" else npl.all_(el, x)
    return cast(el.as_scalar(res), WeakBool)  # the builtins return Python bools
  raise el.error(TypeInferenceError, f"{describe(x)} is not iterable")


@register(any)
def _any(el, x):
  return _anyall(el, x, "any")


@register(all)
def _all(el, x):
  return _anyall(el, x, "all")


@register(sorted)
def _sorted(el, x):
  from . import pure

  return pure.sort1d.__parakeet_impl__(el, el.as_array(x))


@register(isinstance)
def _isinstance(el, x, types):
  def one(t) -> bool:
    if isinstance(t, pytypes.UnionType):
      return builtins.any(one(sub) for sub in t.__args__)
    if isinstance(t, TypeVal):
      if not isinstance(x, Expr):
        return False
      if t.type.weak:  # bool / int / float: Python scalars only
        return x.type.weak and (x.type.kind == t.type.kind or (t.type.kind == "i" and x.type.kind == "b"))
      return x.type is t.type
    if isinstance(t, ComplexTypeVal):
      if isinstance(x, ComplexScalar):
        return x.python_complex if t.weak else (not x.python_complex and x.elt.strong is t.elt.strong)
      return False
    if t is np.ndarray:
      return isinstance(x, (ArrayVal, ComplexArray))
    if t is tuple:
      return isinstance(x, tuple)
    if t is list:
      return isinstance(x, (ListVal, VecVal, FixedListVal))
    if t is dict:
      return isinstance(x, dict)
    if t is range:
      return isinstance(x, RangeVal)
    if t is slice:
      return isinstance(x, slice)
    if t is type(None):
      return x is None
    if t is str:
      return isinstance(x, str)
    if isinstance(t, type) and issubclass(t, (np.integer, np.floating, np.number, np.generic)):
      if not isinstance(x, Expr) or x.type.weak:
        return False
      return issubclass(x.type.dtype.type, t)
    import numbers

    if t in (numbers.Number, numbers.Real):
      return isinstance(x, Expr)
    if t is numbers.Integral:
      return isinstance(x, Expr) and x.type.kind in "iu"
    if isinstance(t, type):
      rec = record_type(x)
      if rec is not None:
        return issubclass(rec.cls, t)
      if isinstance(x, (ArrayVal, Expr, RangeVal, ZipVal, EnumerateVal, TypeVal)) or x is None:
        return False
      return builtins.isinstance(x, t)
    raise el.error(UnsupportedError, f"isinstance() with {describe(t)} is not supported")

  ts = types if isinstance(types, tuple) else (types,)
  return Const(builtins.any(one(t) for t in ts), WeakBool)


_MISSING_ATTR = object()


def _has_attr_static(el, obj, name: str) -> bool:
  from .. import lib
  from ..syntax.records import StructuredRow
  from ..syntax.values import BoundMethod, Closure, Partial, PyFunc

  if isinstance(obj, (np.ufunc, prims.UFunc)):
    return name == "at" or builtins.hasattr(obj, name)
  if isinstance(obj, (ArrayVal, Expr)):
    return lib.lookup_property(name) is not None or lib.lookup_method(obj, name) is not None
  if isinstance(obj, StructuredView):
    return name in ("shape", "ndim", "size", "dtype") or name in obj.names
  if isinstance(obj, StructuredRow):
    return name in obj.names
  if isinstance(obj, RaggedArray):
    if name in ("values", "offsets", "num_segments", "lengths"):
      return True
    return inspect.getattr_static(RaggedArray, name, _MISSING_ATTR) is not _MISSING_ATTR
  rec = record_type(obj)
  if rec is not None:
    if rec.cls is dict:
      return False
    if name in rec.names:
      return True
    if issubclass(rec.cls, tuple) and name == "_replace":
      return True
    if rec.cls.__module__ == "parakeet.index":
      if (rec.cls.__name__ == "SortedIndex" and name in ("get", "range")) or (
        rec.cls.__name__ == "FrozenHashTable" and name == "get"
      ):
        return True
    raw = inspect.getattr_static(rec.cls, name, _MISSING_ATTR)
    return isinstance(raw, (property, pytypes.FunctionType, staticmethod, classmethod))
  if isinstance(obj, TypeVal):
    return name in ("itemsize", "type")
  if isinstance(obj, (tuple, PyFunc, Closure, BoundMethod, Partial, RangeVal)) or obj is None:
    return False
  return builtins.hasattr(obj, name)


@register(hasattr)
def _hasattr(el, obj, name):
  if not isinstance(name, str):
    raise el.error(StaticValueError, f"hasattr() attribute name must be a compile-time string, got {describe(name)}")
  return Const(_has_attr_static(el, obj, name), WeakBool)


_NO_DEFAULT = object()


@register(getattr)
def _getattr_builtin(el, obj, name, default=_NO_DEFAULT):
  if not isinstance(name, str):
    raise el.error(StaticValueError, f"getattr() attribute name must be a compile-time string, got {describe(name)}")
  if _has_attr_static(el, obj, name):
    return el.getattr(obj, name)
  if default is not _NO_DEFAULT:
    return default
  raise el.error(TypeInferenceError, f"{describe(obj)} has no attribute {name!r}")


@register(functools.partial, name="functools.partial")
def _functools_partial(el, fn, *args, **kwargs):
  from ..syntax.values import Partial

  if isinstance(fn, Partial):
    return Partial(fn.fn, (*fn.args, *args), {**fn.kwargs, **kwargs})
  return Partial(fn, tuple(args), dict(kwargs))


# ---------------------------------------------------------------------------
# operator module
# ---------------------------------------------------------------------------


for _op_name, _bin_name in {
  "add": "add",
  "sub": "sub",
  "mul": "mul",
  "truediv": "div",
  "floordiv": "floordiv",
  "mod": "mod",
  "pow": "pow",
  "and_": "bitand",
  "or_": "bitor",
  "xor": "bitxor",
  "lshift": "lshift",
  "rshift": "rshift",
}.items():
  _op_fn = getattr(operator, _op_name)

  def _make_bin(bn):
    def _impl(el, a, b):
      return el.binop(bn, a, b)

    return _impl

  register(_op_fn, name=f"operator.{_op_name}")(_make_bin(_bin_name))


@register(operator.matmul, name="operator.matmul")
def _op_matmul(el, a, b):
  return el.matmul(a, b)


for _cmp_name, _ast_cmp in {
  "eq": "Eq",
  "ne": "NotEq",
  "lt": "Lt",
  "le": "LtE",
  "gt": "Gt",
  "ge": "GtE",
  "is_": "Is",
  "is_not": "IsNot",
}.items():
  import ast as _ast

  _cmp_cls = getattr(_ast, _ast_cmp)

  def _make_cmp(ccls):
    def _impl(el, a, b):
      return el.compare(ccls(), a, b)

    return _impl

  register(getattr(operator, _cmp_name), name=f"operator.{_cmp_name}")(_make_cmp(_cmp_cls))


@register(operator.neg, name="operator.neg")
def _op_neg(el, a):
  return el.apply_ufunc(UF["neg"], [a], python=True)


@register(operator.pos, name="operator.pos")
def _op_pos(el, a):
  return el.apply_ufunc(UF["pos"], [a], python=True)


@register(operator.abs, name="operator.abs")
def _op_abs(el, a):
  return el.apply_ufunc(UF["abs"], [a], python=True)


@register(operator.invert, name="operator.invert")
def _op_invert(el, a):
  return el.apply_ufunc(UF["invert"], [a], python=True)


@register(operator.not_, name="operator.not_")
def _op_not(el, a):
  return ops.lnot(el.truth(a))


@register(operator.truth, name="operator.truth")
def _op_truth(el, a):
  return el.truth(a)


@register(operator.index, name="operator.index")
def _op_index(el, a):
  return el.as_index(a, "index")


@register(operator.getitem, name="operator.getitem")
def _op_getitem(el, a, b):
  return el.getitem(a, b)


@register(operator.itemgetter, name="operator.itemgetter")
def _op_itemgetter(el, *items):
  if not items:
    raise el.error(TypeInferenceError, "itemgetter expected 1 argument, got 0")

  def _get(el_, obj, _items=items):
    if len(_items) == 1:
      return el_.getitem(obj, _items[0])
    return tuple(el_.getitem(obj, it) for it in _items)

  from .adverbs import LibFunction

  return LibFunction("itemgetter", _get)


@register(operator.attrgetter, name="operator.attrgetter")
def _op_attrgetter(el, *attrs):
  if not attrs:
    raise el.error(TypeInferenceError, "attrgetter expected 1 argument, got 0")
  if not builtins.all(isinstance(a, str) for a in attrs):
    raise el.error(StaticValueError, "attrgetter() attribute names must be compile-time strings")

  def _get_one(el_, obj, path: str):
    cur = obj
    for part in path.split("."):
      cur = el_.getattr(cur, part)
    return cur

  def _get(el_, obj, _attrs=attrs):
    if len(_attrs) == 1:
      return _get_one(el_, obj, _attrs[0])
    return tuple(_get_one(el_, obj, a) for a in _attrs)

  from .adverbs import LibFunction

  return LibFunction("attrgetter", _get)


@register(print)
def _print(el, *args, **kwargs):
  raise el.error(
    UnsupportedError,
    "print() is not available in compiled code",
    hint="return the value and print it from Python, or debug with @jit(backend='python')",
  )


# ---------------------------------------------------------------------------
# itertools
# ---------------------------------------------------------------------------


@register(itertools.pairwise, name="itertools.pairwise")
def _it_pairwise(el, iterable):
  if isinstance(iterable, ListVal):
    iterable = tuple(iterable.items)
  if isinstance(iterable, tuple):
    return tuple(builtins.zip(iterable[:-1], iterable[1:]))
  arr = el.as_array(iterable, "itertools.pairwise() argument")
  if arr.rank == 0:
    raise el.error(TypeInferenceError, "itertools.pairwise() requires a 1-d or higher array")
  left = el.getitem(arr, slice(None, Const(-1, I), None))
  right = el.getitem(arr, slice(ops.ONE, None, None))
  return ZipVal((left, right), ops.FALSE)


@register(itertools.islice, name="itertools.islice")
def _it_islice(el, iterable, *args):
  if len(args) == 1:
    start, stop, step = None, args[0], None
  elif len(args) == 2:
    start, stop, step = args[0], args[1], None
  elif len(args) == 3:
    start, stop, step = args[0], args[1], args[2]
  else:
    raise el.error(TypeInferenceError, f"islice expected 2 to 4 arguments, got {len(args) + 1}")
  if isinstance(iterable, ListVal):
    iterable = tuple(iterable.items)
  if isinstance(iterable, tuple):
    s0 = None if start is None else el.static_int(start, "islice start")
    s1 = None if stop is None else el.static_int(stop, "islice stop")
    s2 = None if step is None else el.static_int(step, "islice step")
    return iterable[slice(s0, s1, s2)]
  if isinstance(iterable, RangeVal):
    arr = el.range_array(iterable)
    return el.getitem(arr, slice(start, stop, step))
  arr = el.as_array(iterable, "itertools.islice() argument")
  return el.getitem(arr, slice(start, stop, step))


@register(itertools.chain, name="itertools.chain")
def _it_chain(el, *iterables):
  if builtins.all(isinstance(it, (tuple, ListVal)) for it in iterables):
    out: list = []
    for it in iterables:
      out.extend(it.items if isinstance(it, ListVal) else it)
    return tuple(out)
  return ChainVal(tuple(iterables))


@register(itertools.product, name="itertools.product")
def _it_product(el, *iterables, repeat=1):
  rep = repeat if isinstance(repeat, int) else el.static_int(repeat, "product repeat")
  if rep < 0:
    raise el.error(TypeInferenceError, "repeat argument cannot be negative")
  pools = tuple(iterables) * rep
  if not pools:
    return ((),)
  if builtins.all(isinstance(p, (tuple, ListVal)) for p in pools):
    tpools = [tuple(p.items) if isinstance(p, ListVal) else p for p in pools]
    return tuple(itertools.product(*tpools))
  return ProductVal(pools)


if hasattr(itertools, "batched"):

  @register(itertools.batched, name="itertools.batched")
  def _it_batched(el, iterable, n, strict=False):
    strict_val = strict if isinstance(strict, bool) else el.static_bool(strict, "batched strict")
    if isinstance(iterable, ListVal):
      iterable = tuple(iterable.items)
    if isinstance(iterable, tuple):
      step = n if isinstance(n, int) else el.static_int(n, "batched n")
      if step < 1:
        raise el.error(StaticValueError, "n must be at least one")
      if strict_val and len(iterable) % step != 0:
        raise el.error(StaticValueError, "batched(): incomplete batch")
      return tuple(iterable[i : i + step] for i in builtins.range(0, len(iterable), step))
    nv = el.as_index(n, "batched n")
    el.check(cmp("ge", nv, ops.ONE), "ValueError", "n must be at least one")
    if strict_val:
      length, _ = el.iter_source(iterable)
      el.check(cmp("eq", ops.prim("mod", (length, nv), I), ops.ZERO), "ValueError", "batched(): incomplete batch")
    return BatchedVal(iterable, nv)


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def _math_float(name: str, ufname: str | None = None, nin: int = 1):
  spec = UF[ufname or name]

  def impl(el, *args):
    if len(args) != nin:
      raise el.error(TypeInferenceError, f"math.{name}() takes exactly {nin} argument(s) ({len(args)} given)")
    res = el.apply_ufunc(spec, list(args), python=True)
    if isinstance(res, Expr) and res.type.kind == "f":
      return cast(res, WeakFloat)  # math functions return Python floats
    return res

  impl.__name__ = name
  register(getattr(math, name), name=f"math.{name}")(impl)


for _n, _u in {
    "sqrt": None, "exp": None, "expm1": None, "log10": None, "log2": None, "log1p": None,
    "sin": None, "cos": None, "tan": None, "asin": "arcsin", "acos": "arccos", "atan": "arctan",
    "sinh": None, "cosh": None, "tanh": None, "asinh": "arcsinh", "acosh": "arccosh",
    "atanh": "arctanh", "fabs": None, "erf": None, "erfc": None, "lgamma": None, "gamma": "tgamma",
    "degrees": "rad2deg", "radians": "deg2rad", "cbrt": None, "exp2": None,
}.items():  # fmt: skip
  if hasattr(math, _n):
    _math_float(_n, _u)
for _n, _u in {"atan2": "arctan2", "hypot": None, "copysign": None, "fmod": None, "pow": "float_power"}.items():
  _math_float(_n, _u, nin=2)


@register(math.log, name="math.log")
def _math_log(el, x, base=None):
  res = (
    cast(el.as_scalar(el.apply_ufunc(UF["log"], [x], python=True)), WeakFloat)
    if not isinstance(x, ArrayVal)
    else el.apply_ufunc(UF["log"], [x])
  )
  if base is None:
    return res
  return el.binop("div", res, _math_log(el, base))


def _math_int(name: str):
  spec = UF[name]

  def impl(el, x):
    if isinstance(x, ArrayVal):
      return el.apply_ufunc(spec, [x])
    x = el.as_scalar(x, f"argument of math.{name}()")
    if x.type.kind != "f":
      return cast(x, WeakInt)
    x = el.tmp(x)
    el.check(ops.prim("isfinite", (x,), Bool), "ValueError", "cannot convert float NaN or infinity to integer")
    return cast(ops.prim(name, (x,), x.type), WeakInt)

  impl.__name__ = name
  register(getattr(math, name), name=f"math.{name}")(impl)


for _n in ("floor", "ceil", "trunc"):
  _math_int(_n)


def _math_pred(name: str):
  spec = UF[name]

  def impl(el, x):
    return el.apply_ufunc(spec, [x], python=True)

  impl.__name__ = name
  register(getattr(math, name), name=f"math.{name}")(impl)


for _n in ("isnan", "isinf", "isfinite"):
  _math_pred(_n)

_ = (cmp, norm_axis)
