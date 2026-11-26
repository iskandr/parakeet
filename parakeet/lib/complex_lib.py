"""Unboxed complex64 / complex128 arithmetic, math, linear algebra, and array ops.

Complex values lower cleanly onto pairs of real ``float32`` / ``float64``
arrays or scalar expressions (``ComplexArray`` / ``ComplexScalar``) so every
backend (CPU C, multicore, CUDA, Apple Metal) and automatic differentiation
(Wirtinger ``grad`` / ``jvp`` / ``vjp``) work natively without IR or backend
changes.
"""

from __future__ import annotations

import builtins

import numpy as np

from .. import prims
from ..errors import TypeInferenceError
from ..ndtypes import Float32, Float64, ScalarType, WeakFloat
from ..syntax.ir import Const, Expr
from ..syntax.ops import cast
from ..syntax.records import ComplexArray, ComplexScalar, ComplexTypeVal
from ..syntax.values import ArrayVal, describe
from .core import method, prop, register

UF = prims.UFUNCS


def is_complex(v) -> bool:
  return isinstance(v, (ComplexArray, ComplexScalar))


def is_complex_dtype(dt) -> bool:
  if isinstance(dt, ComplexTypeVal):
    return True
  if dt is complex:
    return True
  if isinstance(dt, np.dtype) and dt.kind == "c":
    return True
  if isinstance(dt, type) and issubclass(dt, np.complexfloating):
    return True
  return False


def complex_float_elt(el, dt) -> ScalarType:
  if isinstance(dt, ComplexTypeVal):
    return dt.elt.strong
  if dt is complex:
    return Float64
  if isinstance(dt, np.dtype) and dt.kind == "c":
    return Float32 if dt.itemsize <= 8 else Float64
  if isinstance(dt, type) and issubclass(dt, np.complexfloating):
    return Float32 if np.dtype(dt).itemsize <= 8 else Float64
  raise el.error(TypeInferenceError, f"not a complex dtype: {dt!r}")


def _real_float_elt(t: ScalarType) -> ScalarType:
  """Component float type promoted from a real ScalarType when combined with complex."""
  if t.weak:
    return WeakFloat
  if t.name == "float32":
    return Float32
  if t.name in ("bool", "int8", "int16", "uint8", "uint16"):
    return Float32
  return Float64


def promote_complex_elt(*vals) -> ScalarType:
  """NEP 50 float component promotion for complex operations (Float32 vs Float64)."""
  has_f32 = False
  has_f64 = False
  for v in vals:
    if isinstance(v, ComplexArray):
      if v.elt.name == "float32":
        has_f32 = True
      else:
        has_f64 = True
    elif isinstance(v, ComplexScalar):
      if v.re.type.weak:
        continue
      if v.re.type.name == "float32":
        has_f32 = True
      else:
        has_f64 = True
    elif isinstance(v, ArrayVal):
      ft = _real_float_elt(v.elt)
      if ft is Float32:
        has_f32 = True
      else:
        has_f64 = True
    elif isinstance(v, Expr):
      if v.type.weak:
        continue
      ft = _real_float_elt(v.type)
      if ft is Float32:
        has_f32 = True
      else:
        has_f64 = True
  if has_f64:
    return Float64
  if has_f32:
    return Float32
  return WeakFloat


def _cast_part(el, v, elt: ScalarType):
  if isinstance(v, ArrayVal):
    return el.cast_array(v, elt) if v.elt is not elt else v
  return cast(v, elt)


def to_parts(el, v, elt: ScalarType) -> tuple[ArrayVal | Expr, ArrayVal | Expr]:
  """Extract ``(re, im)`` cast to ``elt`` from any complex or real scalar/array."""
  if isinstance(v, ComplexArray):
    return _cast_part(el, v.re, elt), _cast_part(el, v.im, elt)
  if isinstance(v, ComplexScalar):
    return cast(v.re, elt), cast(v.im, elt)
  if isinstance(v, ArrayVal):
    re = _cast_part(el, v, elt)
    im = el.delayed(elt, re.shape, lambda _idx, _z=Const(0.0, elt): _z, cost=0)
    return re, im
  if isinstance(v, Expr):
    return cast(v, elt), Const(0.0, elt)
  arr = el.as_array(v, "complex operand")
  re = _cast_part(el, arr, elt)
  im = el.delayed(elt, re.shape, lambda _idx, _z=Const(0.0, elt): _z, cost=0)
  return re, im


def make_complex(el, re, im, python_complex: bool = False):
  if isinstance(re, ArrayVal) or isinstance(im, ArrayVal):
    re_a = re if isinstance(re, ArrayVal) else el.scalar_to_array(re)
    im_a = im if isinstance(im, ArrayVal) else el.scalar_to_array(im)
    if re_a.shape != im_a.shape:
      shape = el.broadcast_shapes([re_a.shape, im_a.shape])
      re_a = el.broadcast_to(re_a, shape)
      im_a = el.broadcast_to(im_a, shape)
    return ComplexArray(re_a, im_a)
  return ComplexScalar(re, im, python_complex=python_complex)


def _ew(el, op: str, a, b):
  py = isinstance(a, Expr) and isinstance(b, Expr) and a.type.weak and b.type.weak
  return el.apply_ufunc(UF[op], [a, b], python=py)


def _ew1(el, op: str, a):
  py = isinstance(a, Expr) and a.type.weak
  return el.apply_ufunc(UF[op], [a], python=py)


def complex_unary(el, op: str, z):
  if op == "pos":
    return z
  if op == "neg":
    return make_complex(el, _ew1(el, "neg", z.re), _ew1(el, "neg", z.im), getattr(z, "python_complex", False))
  raise el.error(TypeInferenceError, f"unsupported unary operator '{op}' for {describe(z)}")


def complex_binop(el, op: str, a, b):
  elt = promote_complex_elt(a, b)
  py_c = (
    isinstance(a, ComplexScalar)
    and a.python_complex
    and (not isinstance(b, ComplexArray))
    and (not isinstance(b, ArrayVal))
    and (not isinstance(b, ComplexScalar) or b.python_complex)
  ) or (
    isinstance(b, ComplexScalar)
    and b.python_complex
    and (not isinstance(a, ComplexArray))
    and (not isinstance(a, ArrayVal))
    and (not isinstance(a, ComplexScalar) or a.python_complex)
  )
  ar, ai = to_parts(el, a, elt)
  br, bi = to_parts(el, b, elt)

  if op == "add":
    return make_complex(el, _ew(el, "add", ar, br), _ew(el, "add", ai, bi), py_c)
  if op == "sub":
    return make_complex(el, _ew(el, "sub", ar, br), _ew(el, "sub", ai, bi), py_c)
  if op == "mul":
    re = _ew(el, "sub", _ew(el, "mul", ar, br), _ew(el, "mul", ai, bi))
    im = _ew(el, "add", _ew(el, "mul", ar, bi), _ew(el, "mul", ai, br))
    return make_complex(el, re, im, py_c)
  if op == "div":
    denom = _ew(el, "add", _ew(el, "mul", br, br), _ew(el, "mul", bi, bi))
    re = _ew(el, "div", _ew(el, "add", _ew(el, "mul", ar, br), _ew(el, "mul", ai, bi)), denom)
    im = _ew(el, "div", _ew(el, "sub", _ew(el, "mul", ai, br), _ew(el, "mul", ar, bi)), denom)
    return make_complex(el, re, im, py_c)
  if op == "pow":
    # Fast exact integer powers (0, 1, 2, 3, -1) and general exp(b * log(a))
    if isinstance(b, Const) and b.type.is_int:
      k = int(b.value)
      if k == 0:
        one = Const(1.0, elt)
        zero = Const(0.0, elt)
        if isinstance(ar, ArrayVal):
          return make_complex(
            el,
            el.delayed(elt, ar.shape, lambda _i, _v=one: _v, cost=0),
            el.delayed(elt, ar.shape, lambda _i, _v=zero: _v, cost=0),
          )
        return make_complex(el, one, zero, py_c)
      if k == 1:
        return make_complex(el, ar, ai, py_c)
      if k == 2:
        return complex_binop(el, "mul", a, a)
      if k == 3:
        return complex_binop(el, "mul", complex_binop(el, "mul", a, a), a)
      if k == -1:
        return complex_binop(el, "div", Const(1.0, elt), a)
    return complex_exp(el, complex_binop(el, "mul", b, complex_log(el, a)))
  if op == "eq":
    return _ew(el, "logical_and", _ew(el, "eq", ar, br), _ew(el, "eq", ai, bi))
  if op == "ne":
    return _ew(el, "logical_or", _ew(el, "ne", ar, br), _ew(el, "ne", ai, bi))
  raise el.error(TypeInferenceError, f"operator '{op}' is not supported for complex operands")


def complex_abs(el, z):
  re, im = z.re, z.im
  return _ew1(el, "sqrt", _ew(el, "add", _ew(el, "mul", re, re), _ew(el, "mul", im, im)))


def complex_conj(el, z):
  if isinstance(z, (ComplexArray, ComplexScalar)):
    return make_complex(el, z.re, _ew1(el, "neg", z.im), getattr(z, "python_complex", False))
  return z


def complex_real(el, z):
  if isinstance(z, (ComplexArray, ComplexScalar)):
    return z.re
  return z


def complex_imag(el, z):
  if isinstance(z, (ComplexArray, ComplexScalar)):
    return z.im
  if isinstance(z, ArrayVal):
    return el.delayed(z.elt, z.shape, lambda _i, _v=Const(0, z.elt): _v, cost=0)
  if isinstance(z, Expr):
    return Const(0, z.type)
  return Const(0.0, WeakFloat)


def complex_angle(el, z):
  if isinstance(z, (ComplexArray, ComplexScalar)):
    return _ew(el, "arctan2", z.im, z.re)
  zero = Const(0.0, Float64)
  return _ew(el, "arctan2", zero, z)


def complex_exp(el, z):
  # exp(a + bi) = exp(a) * (cos(b) + i sin(b))
  er = _ew1(el, "exp", z.re)
  return make_complex(
    el,
    _ew(el, "mul", er, _ew1(el, "cos", z.im)),
    _ew(el, "mul", er, _ew1(el, "sin", z.im)),
    getattr(z, "python_complex", False),
  )


def complex_log(el, z):
  # log(a + bi) = log(|z|) + i * atan2(b, a)
  mag = complex_abs(el, z)
  ang = _ew(el, "arctan2", z.im, z.re)
  return make_complex(el, _ew1(el, "log", mag), ang, getattr(z, "python_complex", False))


def complex_sqrt(el, z):
  # Principal branch: u = sqrt(0.5 * (|z| + re)), v = copysign(sqrt(0.5 * (|z| - re)), im)
  elt = z.re.elt if isinstance(z, ComplexArray) else z.re.type.strong
  half = Const(0.5, elt)
  zero = Const(0.0, elt)
  r = complex_abs(el, z)
  u = _ew1(el, "sqrt", _ew(el, "max", zero, _ew(el, "mul", half, _ew(el, "add", r, z.re))))
  v_mag = _ew1(el, "sqrt", _ew(el, "max", zero, _ew(el, "mul", half, _ew(el, "sub", r, z.re))))
  from . import numpy_lib as npl

  v = npl.where(el, _ew(el, "lt", z.im, zero), _ew1(el, "neg", v_mag), v_mag)
  return make_complex(el, u, v, getattr(z, "python_complex", False))


def complex_sin(el, z):
  # sin(a + bi) = sin(a)cosh(b) + i cos(a)sinh(b)
  re = _ew(el, "mul", _ew1(el, "sin", z.re), _ew1(el, "cosh", z.im))
  im = _ew(el, "mul", _ew1(el, "cos", z.re), _ew1(el, "sinh", z.im))
  return make_complex(el, re, im, getattr(z, "python_complex", False))


def complex_cos(el, z):
  # cos(a + bi) = cos(a)cosh(b) - i sin(a)sinh(b)
  re = _ew(el, "mul", _ew1(el, "cos", z.re), _ew1(el, "cosh", z.im))
  im = _ew1(el, "neg", _ew(el, "mul", _ew1(el, "sin", z.re), _ew1(el, "sinh", z.im)))
  return make_complex(el, re, im, getattr(z, "python_complex", False))


def complex_square(el, z):
  return complex_binop(el, "mul", z, z)


def complex_matmul(el, a, b, what: str = "matmul"):
  from . import numpy_lib as npl

  elt = promote_complex_elt(a, b)
  ar, ai = to_parts(el, a, elt)
  br, bi = to_parts(el, b, elt)
  if not isinstance(ar, ArrayVal) or not isinstance(br, ArrayVal):
    raise el.error(TypeInferenceError, f"{what} requires array operands")
  re = _ew(el, "sub", npl.matmul(el, ar, br, what), npl.matmul(el, ai, bi, what))
  im = _ew(el, "add", npl.matmul(el, ar, bi, what), npl.matmul(el, ai, br, what))
  return make_complex(el, re, im)


def complex_dot(el, a, b):
  from . import numpy_lib as npl

  elt = promote_complex_elt(a, b)
  ar, ai = to_parts(el, a, elt)
  br, bi = to_parts(el, b, elt)
  if not isinstance(ar, ArrayVal) or not isinstance(br, ArrayVal):
    return complex_binop(el, "mul", a, b)
  re = _ew(el, "sub", npl.dot(el, ar, br), npl.dot(el, ai, bi))
  im = _ew(el, "add", npl.dot(el, ar, bi), npl.dot(el, ai, br))
  return make_complex(el, re, im)


def complex_vdot(el, a, b):
  from . import numpy_lib as npl

  prod = complex_binop(el, "mul", complex_conj(el, a), b)
  if isinstance(prod, ComplexArray):
    return make_complex(el, npl.sum_(el, prod.re), npl.sum_(el, prod.im))
  return prod


def complex_astype(el, a, dtype):
  if is_complex_dtype(dtype):
    elt = complex_float_elt(el, dtype)
    re, im = to_parts(el, a, elt)
    return make_complex(el, re, im)
  # Casting complex -> real takes the real part (matching NumPy)
  target = el.as_dtype(dtype)
  re = a.re
  return el.cast_array(re, target) if isinstance(re, ArrayVal) else cast(re, target)


def complex_zeros(el, shape, dtype):
  from . import numpy_lib as npl

  elt = complex_float_elt(el, dtype)
  return ComplexArray(npl._full(el, shape, None, elt, zero=True), npl._full(el, shape, None, elt, zero=True))


def complex_ones(el, shape, dtype):
  from . import numpy_lib as npl

  elt = complex_float_elt(el, dtype)
  return ComplexArray(npl._full(el, shape, Const(1.0, elt), elt), npl._full(el, shape, None, elt, zero=True))


def complex_empty(el, shape, dtype):
  from . import numpy_lib as npl

  elt = complex_float_elt(el, dtype)
  return ComplexArray(npl._full(el, shape, None, elt), npl._full(el, shape, None, elt))


def complex_full(el, shape, fill_value, dtype=None):
  from . import numpy_lib as npl

  elt = complex_float_elt(el, dtype) if dtype is not None else promote_complex_elt(fill_value)
  re, im = to_parts(el, fill_value, elt)
  if isinstance(re, ArrayVal) or isinstance(im, ArrayVal):
    raise el.error(TypeInferenceError, "full() fill_value must be a scalar")
  return ComplexArray(npl._full(el, shape, re, elt), npl._full(el, shape, im, elt))


def complex_sum(el, a: ComplexArray, axis=None, dtype=None, keepdims=False, initial=None):
  from . import numpy_lib as npl

  elt = complex_float_elt(el, dtype) if (dtype is not None and is_complex_dtype(dtype)) else a.elt
  re_a, im_a = to_parts(el, a, elt)
  init_re = init_im = None
  if initial is not None:
    init_re, init_im = to_parts(el, initial, elt)
  r = npl.sum_(el, re_a, axis=axis, keepdims=keepdims, initial=init_re)
  i = npl.sum_(el, im_a, axis=axis, keepdims=keepdims, initial=init_im)
  return make_complex(el, r, i)


def complex_mean(el, a: ComplexArray, axis=None, dtype=None, keepdims=False):
  from . import numpy_lib as npl

  elt = complex_float_elt(el, dtype) if (dtype is not None and is_complex_dtype(dtype)) else a.elt
  re_a, im_a = to_parts(el, a, elt)
  r = npl.mean(el, re_a, axis=axis, keepdims=keepdims)
  i = npl.mean(el, im_a, axis=axis, keepdims=keepdims)
  return make_complex(el, r, i)


def complex_cumsum(el, a: ComplexArray, axis=None, dtype=None):
  from . import numpy_lib as npl

  elt = complex_float_elt(el, dtype) if (dtype is not None and is_complex_dtype(dtype)) else a.elt
  re_a, im_a = to_parts(el, a, elt)
  return make_complex(el, npl.cumsum(el, re_a, axis=axis), npl.cumsum(el, im_a, axis=axis))


def complex_where(el, cond, x, y):
  from . import numpy_lib as npl

  elt = promote_complex_elt(x, y)
  xr, xi = to_parts(el, x, elt)
  yr, yi = to_parts(el, y, elt)
  return make_complex(el, npl.where(el, cond, xr, yr), npl.where(el, cond, xi, yi))


def complex_concatenate(el, arrays, axis=0, dtype=None):
  from . import numpy_lib as npl

  elt = (
    complex_float_elt(el, dtype) if (dtype is not None and is_complex_dtype(dtype)) else promote_complex_elt(*arrays)
  )
  res = []
  ims = []
  for a in arrays:
    r, i = to_parts(el, a, elt)
    res.append(r if isinstance(r, ArrayVal) else el.scalar_to_array(r))
    ims.append(i if isinstance(i, ArrayVal) else el.scalar_to_array(i))
  return ComplexArray(npl.concatenate(el, tuple(res), axis=axis), npl.concatenate(el, tuple(ims), axis=axis))


def complex_stack(el, arrays, axis=0, dtype=None):
  from . import numpy_lib as npl

  elt = (
    complex_float_elt(el, dtype) if (dtype is not None and is_complex_dtype(dtype)) else promote_complex_elt(*arrays)
  )
  res = []
  ims = []
  for a in arrays:
    r, i = to_parts(el, a, elt)
    res.append(r if isinstance(r, ArrayVal) else el.scalar_to_array(r))
    ims.append(i if isinstance(i, ArrayVal) else el.scalar_to_array(i))
  return ComplexArray(npl.stack(el, tuple(res), axis=axis), npl.stack(el, tuple(ims), axis=axis))


# ---------------------------------------------------------------------------
# Registered Python / NumPy complex functions, properties, and methods
# ---------------------------------------------------------------------------


@register(builtins.complex, name="complex")
def _complex_ctor(el, real=None, imag=None):
  if real is None:
    return ComplexScalar(Const(0.0, WeakFloat), Const(0.0, WeakFloat), python_complex=True)
  if isinstance(real, (ComplexArray, ComplexScalar)):
    if imag is not None:
      return complex_binop(
        el,
        "add",
        real,
        complex_binop(el, "mul", ComplexScalar(Const(0.0, WeakFloat), Const(1.0, WeakFloat), True), imag),
      )
    return real
  rs = el.as_scalar(real, "complex() real part")
  is_weak = rs.type.weak and (imag is None or el.as_scalar(imag, "complex() imag part").type.weak)
  ft = WeakFloat if is_weak else Float64
  r = cast(rs, ft)
  i = Const(0.0, ft) if imag is None else cast(el.as_scalar(imag, "complex() imag part"), ft)
  return ComplexScalar(r, i, python_complex=True)


def call_complex_type(el, tv: ComplexTypeVal, args: list, kwargs: dict):
  if tv.weak:
    return _complex_ctor(el, *args, **kwargs)
  if tv.elt is Float32:
    return _np_complex64(el, *args, **kwargs)
  return _np_complex128(el, *args, **kwargs)


@register(np.complex64, name="np.complex64")
def _np_complex64(el, x=None):
  if x is None:
    return ComplexScalar(Const(0.0, Float32), Const(0.0, Float32))
  re, im = to_parts(el, x, Float32)
  return make_complex(el, re, im)


@register(np.complex128, name="np.complex128")
def _np_complex128(el, x=None):
  if x is None:
    return ComplexScalar(Const(0.0, Float64), Const(0.0, Float64))
  re, im = to_parts(el, x, Float64)
  return make_complex(el, re, im)


@register(np.real, name="np.real")
def _np_real(el, val):
  return complex_real(el, val)


@register(np.imag, name="np.imag")
def _np_imag(el, val):
  return complex_imag(el, val)


@register(np.conj, np.conjugate, name="np.conj")
def _np_conj(el, x):
  return complex_conj(el, x)


@register(np.angle, name="np.angle")
def _np_angle(el, z, deg=False):
  ang = complex_angle(el, z)
  if deg is not False and el.static_bool(deg, "deg"):
    elt = ang.elt if isinstance(ang, ArrayVal) else ang.type.strong
    return _ew(el, "mul", ang, Const(180.0 / np.pi, elt))
  return ang


@prop("real")
def _prop_real(el, x):
  return complex_real(el, x)


@prop("imag")
def _prop_imag(el, x):
  return complex_imag(el, x)


@method("conj", "conjugate")
def _method_conj(el, x):
  return complex_conj(el, x)
