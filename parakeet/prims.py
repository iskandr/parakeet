"""Typing and lowering rules for scalar (element-wise) operations.

Each :class:`UFunc` knows the NumPy ufunc that defines its typing (so result
dtypes match NumPy exactly) and how to lower itself to IR primitives.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .ndtypes import (
  Bool,
  Float32,
  Float64,
  Int32,
  ScalarType,
  UInt8,
  UInt16,
  WeakBool,
  WeakFloat,
  WeakInt,
  resolve_ufunc,
)
from .syntax import ops
from .syntax.ir import Const, Expr
from .syntax.ops import cast, prim


@dataclass(frozen=True, eq=False)
class UFunc:
  name: str
  nin: int
  np: np.ufunc | None  # typing oracle
  kind: str
  emit: Callable[[tuple[Expr, ...], ScalarType], Expr]
  cost: int = 1
  py: Callable | None = None

  def __repr__(self) -> str:
    return f"<parakeet ufunc {self.name}>"

  @property
  def __name__(self) -> str:
    return self.name

  def __call__(self, *args, **kwargs):
    """Outside compiled code these behave like the NumPy ufunc they model."""
    if self.py is not None:
      return self.py(*args, **kwargs)
    if self.np is None:
      import math

      return getattr(math, {"tgamma": "gamma"}.get(self.name, self.name))(*args)
    return self.np(*args, **kwargs)


UFUNCS: dict[str, UFunc] = {}
BY_NUMPY: dict[np.ufunc, UFunc] = {}


def _register(name: str, nin: int, npf, kind: str, emit, cost: int = 1, py=None) -> UFunc:
  uf = UFunc(name, nin, npf, kind, emit, cost, py)
  UFUNCS[name] = uf
  if npf is not None and npf not in BY_NUMPY:
    BY_NUMPY[npf] = uf
  return uf


def _direct(op: str):
  def emit(xs, out):
    return prim(op, xs, out)

  return emit


# ---- arithmetic -------------------------------------------------------------


def _add(xs, out):
  return prim("or" if out.kind == "b" else "add", xs, out)


def _mul(xs, out):
  return prim("and" if out.kind == "b" else "mul", xs, out)


def _abs(xs, out):
  return xs[0] if out.kind in "ub" else prim("abs", xs, out)


def _pow(xs, out):
  base, exp = xs
  if isinstance(exp, Const) and out.kind == "f":
    if exp.value == 2.0:
      return prim("mul", (base, base), out)
    if exp.value == 1.0:
      return base
    if exp.value == 0.5:
      return prim("sqrt", (base,), out)
  if isinstance(exp, Const) and out.kind in "iu" and exp.value == 2:
    return prim("mul", (base, base), out)
  return prim("pow", xs, out)


_register("add", 2, np.add, "arith", _add)
_register("sub", 2, np.subtract, "arith", _direct("sub"))
_register("mul", 2, np.multiply, "arith", _mul)
_register("div", 2, np.true_divide, "div", _direct("div"))
_register("floordiv", 2, np.floor_divide, "intdiv", _direct("floordiv"), cost=2)
_register("mod", 2, np.remainder, "intdiv", _direct("mod"), cost=2)
_register("fmod", 2, np.fmod, "arith", _direct("fmod"), cost=2)
_register("pow", 2, np.power, "pow", _pow, cost=3)
_register("float_power", 2, np.float_power, "float", _pow, cost=3)
_register("neg", 1, np.negative, "neg", _direct("neg"))
_register("pos", 1, np.positive, "neg", lambda xs, out: xs[0])
_register("abs", 1, np.absolute, "neg", _abs)
_register("fabs", 1, np.fabs, "float", _direct("abs"))


def _minmax(op, bool_op):
  def emit(xs, out):
    return prim(bool_op if out.kind == "b" else op, xs, out)

  return emit


_register("min", 2, np.minimum, "minmax", _minmax("min", "and"))
_register("max", 2, np.maximum, "minmax", _minmax("max", "or"))


def _fminmax(op):
  def emit(xs, out):
    a, b = xs
    if out.kind == "b":
      return prim("and" if op == "min" else "or", xs, out)
    if out.kind != "f":
      return prim(op, xs, out)
    nan_a = prim("isnan", (a,), Bool)
    nan_b = prim("isnan", (b,), Bool)
    return ops.select(nan_a, b, ops.select(nan_b, a, prim(op, xs, out)))

  return emit


_register("fmin", 2, np.fmin, "minmax", _fminmax("min"), cost=2)
_register("fmax", 2, np.fmax, "minmax", _fminmax("max"), cost=2)

# ---- comparisons ------------------------------------------------------------

for _name, _npf in ops.COMPARE.items():
  _register(_name, 2, _npf, "cmp", _direct(_name))

# ---- logical / bitwise ------------------------------------------------------


def _truthy(x: Expr) -> Expr:
  return cast(x, Bool)


def _logical(op):
  def emit(xs, out):
    return prim(op, tuple(_truthy(x) for x in xs), out)

  return emit


_register("logical_and", 2, np.logical_and, "logical", _logical("and"))
_register("logical_or", 2, np.logical_or, "logical", _logical("or"))
_register("logical_not", 1, np.logical_not, "logical", _logical("not"))
_register(
    "logical_xor", 2, np.logical_xor, "logical",
    lambda xs, out: prim("ne", (_truthy(xs[0]), _truthy(xs[1])), out),
)  # fmt: skip


def _bit(int_op, bool_op):
  def emit(xs, out):
    return prim(bool_op if out.kind == "b" else int_op, xs, out)

  return emit


_register("bitand", 2, np.bitwise_and, "bit", _bit("bitand", "and"))
_register("bitor", 2, np.bitwise_or, "bit", _bit("bitor", "or"))
_register("bitxor", 2, np.bitwise_xor, "bit", _bit("bitxor", "ne"))
_register("invert", 1, np.invert, "invert", _bit("invert", "not"))
_register("lshift", 2, np.left_shift, "shift", _direct("lshift"))
_register("rshift", 2, np.right_shift, "shift", _direct("rshift"))


def _py_bit_scalar(op: str, x, k=None):
  if isinstance(x, bool):
    bits = 8
    u = int(x) & 0xFF
  elif isinstance(x, int):
    bits = 64
    u = int(x) & ((1 << 64) - 1)
  elif isinstance(x, np.integer):
    bits = x.dtype.itemsize * 8
    u = int(x) & ((1 << bits) - 1)
  else:
    raise TypeError(f"{op}() requires integer inputs")
  if op == "popcount":
    return u.bit_count()
  if op == "clz":
    return bits if u == 0 else bits - u.bit_length()
  if op == "ctz":
    return bits if u == 0 else (u & -u).bit_length() - 1
  if op == "rotl":
    r = int(k) & (bits - 1)
    res = u if r == 0 else ((u << r) | (u >> (bits - r))) & ((1 << bits) - 1)
    return type(x)(np.array(res, dtype=np.uint64).astype(x.dtype)) if isinstance(x, np.integer) else res
  if op == "rotr":
    r = int(k) & (bits - 1)
    res = u if r == 0 else ((u >> r) | (u << (bits - r))) & ((1 << bits) - 1)
    return type(x)(np.array(res, dtype=np.uint64).astype(x.dtype)) if isinstance(x, np.integer) else res
  if op == "bswap":
    res = int.from_bytes(u.to_bytes(bits // 8, "little"), "big")
    return type(x)(np.array(res, dtype=np.uint64).astype(x.dtype)) if isinstance(x, np.integer) else res
  raise ValueError(op)


def _py_bitop(op: str):
  def fn(x, k=None):
    if op in ("rotl", "rotr"):
      if isinstance(x, np.ndarray) or isinstance(k, np.ndarray):
        xa, ka = np.broadcast_arrays(np.asarray(x), np.asarray(k))
        bits = xa.dtype.itemsize * 8
        u = xa.view(np.dtype(f"u{xa.dtype.itemsize}")).astype(np.uint64)
        r = (ka.astype(np.int64) & (bits - 1)).astype(np.uint64)
        mask = np.uint64((1 << bits) - 1)
        if op == "rotl":
          res = np.where(r == 0, u, ((u << r) | (u >> (np.uint64(bits) - r))) & mask)
        else:
          res = np.where(r == 0, u, ((u >> r) | (u << (np.uint64(bits) - r))) & mask)
        return res.astype(f"u{xa.dtype.itemsize}").view(xa.dtype)
      return _py_bit_scalar(op, x, k)
    if isinstance(x, np.ndarray):
      if op == "bswap":
        return x.byteswap()
      flat = [_py_bit_scalar(op, v) for v in x.ravel()]
      return np.array(flat, dtype=np.int32).reshape(x.shape)
    res = _py_bit_scalar(op, x)
    if op in ("popcount", "clz", "ctz") and isinstance(x, np.integer):
      return np.int32(res)
    return res

  return fn


def _py_bf16_to_f32(x):
  if isinstance(x, np.ndarray):
    u32 = x.astype(np.uint16).astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)
  u32 = (int(x) & 0xFFFF) << 16
  return np.float32(struct.unpack("<f", struct.pack("<I", u32))[0])


def _py_f32_to_bf16(x):
  if isinstance(x, np.ndarray):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    is_nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    nan_val = ((u >> np.uint32(16)) | np.uint32(0x0040)).astype(np.uint16)
    rne_val = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
    return np.where(is_nan, nan_val, rne_val)
  u = struct.unpack("<I", struct.pack("<f", float(np.float32(x))))[0]
  if (u & 0x7FFFFFFF) > 0x7F800000:
    return np.uint16(((u >> 16) | 0x0040) & 0xFFFF)
  return np.uint16(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF)


def _bitwise_count_emit(xs, out):
  (x,) = xs
  if x.type.kind == "i":
    x = prim("abs", (x,), x.type)
  return cast(prim("popcount", (x,), Int32), out)


_register("popcount", 1, None, "bitcount", _direct("popcount"), py=_py_bitop("popcount"))
_register("clz", 1, None, "bitcount", _direct("clz"), py=_py_bitop("clz"))
_register("ctz", 1, None, "bitcount", _direct("ctz"), py=_py_bitop("ctz"))
_register("rotl", 2, None, "bitrot", _direct("rotl"), py=_py_bitop("rotl"))
_register("rotr", 2, None, "bitrot", _direct("rotr"), py=_py_bitop("rotr"))
_register("bswap", 1, None, "bswap", _direct("bswap"), py=_py_bitop("bswap"))
_register("bf16_to_f32", 1, None, "bf16_to_f32", _direct("bf16_to_f32"), py=_py_bf16_to_f32)
_register("f32_to_bf16", 1, None, "f32_to_bf16", _direct("f32_to_bf16"), py=_py_f32_to_bf16)
if hasattr(np, "bitwise_count"):
  _register("bitwise_count", 1, np.bitwise_count, "bitcount_np", _bitwise_count_emit)

# ---- float functions --------------------------------------------------------


def _float_or_identity(op):
  def emit(xs, out):
    # recent NumPy keeps integers for floor/ceil/trunc/rint of integer input
    return xs[0] if out.kind != "f" else prim(op, xs, out)

  return emit


for _name in ops.FLOAT_UNARY:
  _emit = _float_or_identity(_name) if _name in ("floor", "ceil", "trunc", "rint") else _direct(_name)
  _register(
    _name,
    1,
    ops.FLOAT_UNARY[_name],
    "float",
    _emit,
    cost=1 if _name in ("floor", "ceil", "trunc", "rint", "sqrt") else 4,
  )

_register("arctan2", 2, np.arctan2, "float", _direct("arctan2"), cost=4)
_register("hypot", 2, np.hypot, "float", _direct("hypot"), cost=3)
_register("copysign", 2, np.copysign, "float", _direct("copysign"))
for _name in ops.MATH_ONLY:
  _register(_name, 1, None, "float", _direct(_name), cost=6)


def _pred(op, int_result):
  def emit(xs, out):
    if xs[0].type.kind != "f":
      if op == "signbit":
        if xs[0].type.kind in "ub":
          return Const(False, out)
        return prim("lt", (xs[0], Const(0, xs[0].type)), out)
      return Const(int_result, out)
    return prim(op, xs, out)

  return emit


_register("isnan", 1, np.isnan, "pred", _pred("isnan", False))
_register("isinf", 1, np.isinf, "pred", _pred("isinf", False))
_register("isfinite", 1, np.isfinite, "pred", _pred("isfinite", True))
_register("signbit", 1, np.signbit, "pred", _pred("signbit", False))

# ---- derived ----------------------------------------------------------------


def _sign(xs, out):
  (x,) = xs
  if out.kind == "b":
    return x
  zero, one = Const(0, out), Const(1, out)
  pos = prim("gt", (x, zero), Bool)
  if out.kind == "u":
    return ops.select(pos, one, zero)
  neg = prim("lt", (x, zero), Bool)
  res = ops.select(pos, one, ops.select(neg, Const(-1, out), zero))
  if out.kind == "f":  # sign(nan) is nan
    res = ops.select(prim("isnan", (x,), Bool), x, res)
  return res


def _square(xs, out):
  return prim("and" if out.kind == "b" else "mul", (xs[0], xs[0]), out)


def _reciprocal(xs, out):
  if out.kind == "f":
    return prim("div", (Const(1, out), xs[0]), out)
  return prim("floordiv", (Const(1, out), xs[0]), out)


def _scale(factor):
  def emit(xs, out):
    return prim("mul", (xs[0], Const(factor, out)), out)

  return emit


def _logaddexp(base2: bool):
  def emit(xs, out):
    a, b = xs
    hi = prim("max", (a, b), out)
    lo = prim("min", (a, b), out)
    diff = prim("sub", (lo, hi), out)
    if base2:
      inner = prim("log1p", (prim("exp2", (diff,), out),), out)
      tail = prim("mul", (inner, Const(1.0 / math.log(2.0), out)), out)
    else:
      tail = prim("log1p", (prim("exp", (diff,), out),), out)
    res = prim("add", (hi, tail), out)
    # a == b (including infinities of equal sign) would give inf - inf
    same = prim("eq", (a, b), Bool)
    ln2 = Const(1.0 if base2 else math.log(2.0), out)
    return ops.select(same, prim("add", (a, ln2), out), res)

  return emit


def _heaviside(xs, out):
  x, h = xs
  zero, one = Const(0, out), Const(1, out)
  res = ops.select(prim("lt", (x, zero), Bool), zero, ops.select(prim("gt", (x, zero), Bool), one, h))
  return ops.select(prim("isnan", (x,), Bool), x, res)


_register("sign", 1, np.sign, "neg", _sign, cost=2)
_register("square", 1, np.square, "neg", _square)
_register("reciprocal", 1, np.reciprocal, "neg", _reciprocal, cost=2)
_register("rad2deg", 1, np.rad2deg, "float", _scale(180.0 / math.pi))
_register("deg2rad", 1, np.deg2rad, "float", _scale(math.pi / 180.0))
BY_NUMPY[np.degrees] = UFUNCS["rad2deg"]
BY_NUMPY[np.radians] = UFUNCS["deg2rad"]
_register("logaddexp", 2, np.logaddexp, "float", _logaddexp(False), cost=10)
_register("logaddexp2", 2, np.logaddexp2, "float", _logaddexp(True), cost=10)
_register("heaviside", 2, np.heaviside, "float", _heaviside, cost=3)

for _alias, _target in {
    "abs": np.abs, "div": np.divide, "mod": np.mod, "invert": np.bitwise_not,
    "arcsin": getattr(np, "asin", None), "arccos": getattr(np, "acos", None),
    "arctan": getattr(np, "atan", None), "arcsinh": getattr(np, "asinh", None),
    "arccosh": getattr(np, "acosh", None), "arctanh": getattr(np, "atanh", None),
    "arctan2": getattr(np, "atan2", None), "pow": getattr(np, "pow", None),
    "lshift": getattr(np, "bitwise_left_shift", None),
    "rshift": getattr(np, "bitwise_right_shift", None),
    "invert ": getattr(np, "bitwise_invert", None),
}.items():  # fmt: skip
  if _target is not None and isinstance(_target, np.ufunc):
    BY_NUMPY.setdefault(_target, UFUNCS[_alias.strip()])


# --------------------------------------------------------------------------
# typing
# --------------------------------------------------------------------------

_RANK = {"b": 0, "i": 1, "u": 1, "f": 2}


def resolve_weak(uf: UFunc, types: tuple[ScalarType, ...]):
  """Python semantics for operations whose operands are all Python scalars.

  Returns ``(input_types, output_type)`` or ``None`` to defer to NumPy.
  """
  top = max(_RANK[t.kind] for t in types)
  kind = uf.kind
  n = len(types)
  if kind in ("arith", "intdiv", "pow"):
    out = WeakFloat if top == 2 else WeakInt
    return (out,) * n, out
  if kind == "div" or kind == "float":
    return (WeakFloat,) * n, WeakFloat
  if kind == "cmp":
    common = (WeakBool, WeakInt, WeakFloat)[top]
    return (common,) * n, WeakBool
  if kind == "minmax":
    common = (WeakBool, WeakInt, WeakFloat)[top]
    return (common,) * n, common
  if kind == "neg":
    out = WeakFloat if top == 2 else WeakInt
    return (out,) * n, out
  if kind == "bit":
    if top == 2:
      raise TypeError(f"unsupported operand type(s) for {uf.name}: 'float'")
    out = WeakBool if top == 0 else WeakInt
    return (out,) * n, out
  if kind in ("shift", "invert"):
    if top == 2:
      raise TypeError(f"unsupported operand type(s) for {uf.name}: 'float'")
    return (WeakInt,) * n, WeakInt
  if kind == "logical":
    return (WeakBool,) * n, WeakBool
  if kind == "pred":
    return (WeakFloat,) * n, WeakBool
  return None


def resolve(uf: UFunc, types: tuple[ScalarType, ...], python: bool):
  """Pick input/output types for ``uf`` applied to operands of ``types``.

  ``python`` selects Python-operator semantics when every operand is a weak
  (Python) scalar; otherwise NumPy's own type resolution decides.
  """
  if python and all(t.weak for t in types):
    res = resolve_weak(uf, types)
    if res is not None:
      return res
  if uf.kind == "bitcount":
    t0 = types[0]
    if t0.kind == "f":
      raise TypeError(f"{uf.name} requires integer operands, got {t0.name}")
    if t0.weak:
      return (WeakInt,), WeakInt
    inp = UInt8 if t0.kind == "b" else t0
    return (inp,), Int32
  if uf.kind == "bitrot":
    t0, t1 = types
    if t0.kind == "f" or t1.kind == "f":
      raise TypeError(f"{uf.name} requires integer operands, got {t0.name}, {t1.name}")
    if t0.weak:
      return (WeakInt, Int32), WeakInt
    inp = UInt8 if t0.kind == "b" else t0
    return (inp, Int32), inp
  if uf.kind == "bswap":
    t0 = types[0]
    if t0.kind not in "iu":
      raise TypeError(f"bswap requires integer operands, got {t0.name}")
    return (t0,), t0
  if uf.kind == "bf16_to_f32":
    t0 = types[0]
    if t0.kind not in "iu":
      raise TypeError(f"bf16_to_f32 requires integer operands, got {t0.name}")
    return (UInt16,), Float32
  if uf.kind == "f32_to_bf16":
    return (Float32,), UInt16
  if uf.np is None:
    out = WeakFloat if all(t.weak for t in types) else None
    if out is None:
      small = all(t.name == "float32" or t.weak for t in types)
      out = Float32 if small else Float64
    return (out,) * len(types), out
  ins, outs = resolve_ufunc(uf.np, types)
  return ins, outs[0]


# NumPy-style names for the primitives, usable as first-class functions:
# reduce(prims.add, x)
add = UFUNCS["add"]
subtract = UFUNCS["sub"]
multiply = UFUNCS["mul"]
divide = UFUNCS["div"]
true_divide = UFUNCS["div"]
floor_divide = UFUNCS["floordiv"]
remainder = mod = UFUNCS["mod"]
fmod = UFUNCS["fmod"]
power = UFUNCS["pow"]
negative = UFUNCS["neg"]
minimum = UFUNCS["min"]
maximum = UFUNCS["max"]
equal = UFUNCS["eq"]
not_equal = UFUNCS["ne"]
less = UFUNCS["lt"]
less_equal = UFUNCS["le"]
greater = UFUNCS["gt"]
greater_equal = UFUNCS["ge"]
logical_and = UFUNCS["logical_and"]
logical_or = UFUNCS["logical_or"]
logical_not = UFUNCS["logical_not"]
logical_xor = UFUNCS["logical_xor"]
bitwise_and = UFUNCS["bitand"]
bitwise_or = UFUNCS["bitor"]
bitwise_xor = UFUNCS["bitxor"]
bitwise_not = UFUNCS["invert"]
popcount = UFUNCS["popcount"]
clz = UFUNCS["clz"]
ctz = UFUNCS["ctz"]
rotl = UFUNCS["rotl"]
rotr = UFUNCS["rotr"]
bswap = UFUNCS["bswap"]
bf16_to_f32 = UFUNCS["bf16_to_f32"]
f32_to_bf16 = UFUNCS["f32_to_bf16"]
