"""Staged implementations of Python builtins and the ``math`` module."""

from __future__ import annotations

import builtins
import dataclasses
import math

import numpy as np

from .. import prims
from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..ndtypes import Bool, WeakBool, WeakFloat, WeakInt
from ..syntax import ops
from ..syntax.ir import Const, Expr
from ..syntax.ops import I, cast, cmp
from ..syntax.records import StructuredView, record_type
from ..syntax.values import ArrayVal, EnumerateVal, RangeVal, TypeVal, ZipVal, describe
from . import numpy_lib as npl
from .core import method, norm_axis, reduce_values, register

UF = prims.UFUNCS


@register(len)
def _len(el, x):
  if isinstance(x, ArrayVal):
    if x.rank == 0:
      raise el.error(TypeInferenceError, "len() of a 0-d array")
    return x.shape[0]
  if isinstance(x, StructuredView):
    if x.rank == 0:
      raise el.error(TypeInferenceError, "len() of a 0-d structured array")
    return x.shape[0]
  if isinstance(x, (tuple, str)):
    return Const(len(x), I)
  if isinstance(x, RangeVal):
    return el.range_length(x)
  if isinstance(x, (ZipVal, EnumerateVal)):
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
    raise el.error(TypeInferenceError, f"replace() expects a frozen dataclass instance, got {describe(obj)}")
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
  if isinstance(x, tuple) and not builtins.all(isinstance(v, Expr) for v in x):
    # a sequence of non-scalars (e.g. zip of tuples) has no array form:
    # keep it as an immutable sequence
    ranks = {v.rank if isinstance(v, ArrayVal) else None for v in x}
    if None in ranks or len(ranks) != 1:
      return x
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
    if isinstance(t, TypeVal):
      if not isinstance(x, Expr):
        return False
      if t.type.weak:  # bool / int / float: Python scalars only
        return x.type.weak and (x.type.kind == t.type.kind or (t.type.kind == "i" and x.type.kind == "b"))
      return x.type is t.type
    if t is np.ndarray:
      return isinstance(x, ArrayVal)
    if t is tuple:
      return isinstance(x, tuple)
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
    raise el.error(UnsupportedError, f"isinstance() with {describe(t)} is not supported")

  ts = types if isinstance(types, tuple) else (types,)
  return Const(builtins.any(one(t) for t in ts), WeakBool)


@register(print)
def _print(el, *args, **kwargs):
  raise el.error(
    UnsupportedError,
    "print() is not available in compiled code",
    hint="return the value and print it from Python, or debug with @jit(backend='python')",
  )


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
