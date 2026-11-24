"""Primitive operation catalogue and a constant-folding expression builder."""

from __future__ import annotations

import math
import struct

import numpy as np

from ..ndtypes import Bool, ScalarType, WeakBool, WeakInt
from .ir import Cast, Const, Expr, Prim

I = WeakInt  # type of all shape/index arithmetic (a Python int)  # noqa: E741

# op name -> NumPy function used to fold constants (and as the reference semantics)
ARITH = {
  "add": np.add,
  "sub": np.subtract,
  "mul": np.multiply,
  "div": np.true_divide,
  "floordiv": np.floor_divide,
  "mod": np.remainder,
  "fmod": np.fmod,
  "pow": np.power,
  "min": np.minimum,
  "max": np.maximum,
  "neg": np.negative,
  "abs": np.absolute,
  "bitand": np.bitwise_and,
  "bitor": np.bitwise_or,
  "bitxor": np.bitwise_xor,
  "invert": np.invert,
  "lshift": np.left_shift,
  "rshift": np.right_shift,
  "arctan2": np.arctan2,
  "hypot": np.hypot,
  "copysign": np.copysign,
}
COMPARE = {
  "eq": np.equal,
  "ne": np.not_equal,
  "lt": np.less,
  "le": np.less_equal,
  "gt": np.greater,
  "ge": np.greater_equal,
}
LOGICAL = {"and": np.logical_and, "or": np.logical_or, "not": np.logical_not}
FLOAT_UNARY = {
  name: getattr(np, name)
  for name in (
    "sqrt exp exp2 expm1 log log10 log2 log1p sin cos tan arcsin arccos arctan "
    "sinh cosh tanh arcsinh arccosh arctanh floor ceil trunc rint cbrt"
  ).split()
}
PREDICATES = {
  "isnan": np.isnan,
  "isinf": np.isinf,
  "isfinite": np.isfinite,
  "signbit": np.signbit,
}
# operations without a NumPy ufunc: folded with the math module
MATH_ONLY = {"erf": math.erf, "erfc": math.erfc, "lgamma": math.lgamma, "tgamma": math.gamma}
BITOPS = frozenset({"popcount", "clz", "ctz", "rotl", "rotr", "bswap"})
BF16_OPS = frozenset({"bf16_to_f32", "f32_to_bf16"})

ALL_OPS = (
  set(ARITH) | set(COMPARE) | set(LOGICAL) | set(FLOAT_UNARY) | set(PREDICATES) | set(MATH_ONLY) | BITOPS | BF16_OPS
) | {"select"}

ASSOCIATIVE = frozenset({"add", "mul", "min", "max", "and", "or", "bitand", "bitor", "bitxor"})


def const(value, type: ScalarType = I) -> Const:
  return Const(value, type)


def is_const(e, value=None) -> bool:
  if not isinstance(e, Const):
    return False
  return value is None or e.value == value


TRUE = Const(True, WeakBool)
FALSE = Const(False, WeakBool)
ZERO = Const(0, I)
ONE = Const(1, I)


def _np(c: Const):
  return c.type.dtype.type(c.value)


def _fold_bitop(op: str, args: tuple[Const, ...], type: ScalarType) -> Const:
  t0 = args[0].type
  bits = t0.bits
  mask = (1 << bits) - 1
  u = int(args[0].value) & mask
  if op == "popcount":
    return Const(u.bit_count(), type)
  if op == "clz":
    return Const(bits if u == 0 else bits - u.bit_length(), type)
  if op == "ctz":
    return Const(bits if u == 0 else (u & -u).bit_length() - 1, type)
  if op == "rotl":
    r = int(args[1].value) & (bits - 1)
    res = u if r == 0 else ((u << r) | (u >> (bits - r))) & mask
    return Const(np.array(res, dtype=np.uint64).astype(type.dtype).item(), type)
  if op == "rotr":
    r = int(args[1].value) & (bits - 1)
    res = u if r == 0 else ((u >> r) | (u << (bits - r))) & mask
    return Const(np.array(res, dtype=np.uint64).astype(type.dtype).item(), type)
  if op == "bswap":
    nbytes = bits // 8
    res = int.from_bytes(u.to_bytes(nbytes, "little"), "big")
    return Const(np.array(res, dtype=np.uint64).astype(type.dtype).item(), type)
  raise ValueError(op)


def _fold_bf16(op: str, args: tuple[Const, ...], type: ScalarType) -> Const:
  if op == "bf16_to_f32":
    u16 = int(args[0].value) & 0xFFFF
    f32 = struct.unpack("<f", struct.pack("<I", u16 << 16))[0]
    return Const(np.float32(f32).item(), type)
  f = np.float32(args[0].value)
  u = struct.unpack("<I", struct.pack("<f", float(f)))[0]
  if (u & 0x7FFFFFFF) > 0x7F800000:
    res = ((u >> 16) | 0x0040) & 0xFFFF
  else:
    res = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
  return Const(res, type)


def _fold(op: str, args: tuple[Const, ...], type: ScalarType) -> Const | None:
  t0 = args[0].type
  try:
    if op in MATH_ONLY:
      value = MATH_ONLY[op](args[0].value)
    elif op in BITOPS:
      return _fold_bitop(op, args, type)
    elif op in BF16_OPS:
      return _fold_bf16(op, args, type)
    elif op == "select":
      return Const(args[1].value if args[0].value else args[2].value, type)
    else:
      fn = (
                ARITH.get(op) or COMPARE.get(op) or LOGICAL.get(op)
                or FLOAT_UNARY.get(op) or PREDICATES[op]
            )  # fmt: skip
      if op in ("floordiv", "mod", "fmod") and t0.kind in "iub" and args[1].value == 0:
        return None
      if op == "pow" and t0.kind in "iu" and args[1].value < 0:
        return None
      if op in ("lshift", "rshift") and not 0 <= args[1].value < t0.bits:
        return None
      with np.errstate(all="ignore"):
        value = fn(*[_np(a) for a in args]).item()
  except (ValueError, OverflowError, TypeError, ZeroDivisionError):
    return None
  return Const(value, type)


def prim(op: str, args: tuple[Expr, ...], type: ScalarType) -> Expr:
  """Build ``op(*args)`` of result type ``type``, folding and simplifying."""
  if all(isinstance(a, Const) for a in args):
    folded = _fold(op, args, type)  # type: ignore[arg-type]
    if folded is not None:
      return folded
  if len(args) == 2:
    a, b = args
    integral = type.kind in "iu"
    if op == "add" and integral:
      if is_const(a, 0):
        return retype(b, type)
      if is_const(b, 0):
        return retype(a, type)
    elif op == "sub" and integral:
      if is_const(b, 0):
        return retype(a, type)
    elif op == "mul" and integral:
      if is_const(a, 1):
        return retype(b, type)
      if is_const(b, 1):
        return retype(a, type)
      if is_const(a, 0) or is_const(b, 0):
        return Const(0, type)
    elif op == "floordiv" and integral:
      if is_const(b, 1):
        return retype(a, type)
    elif op == "and":
      if isinstance(a, Const):
        return retype(b, type) if a.value else Const(False, type)
      if isinstance(b, Const) and b.value:
        return retype(a, type)
    elif op == "or":
      if isinstance(a, Const):
        return Const(True, type) if a.value else retype(b, type)
      if isinstance(b, Const) and not b.value:
        return retype(a, type)
  elif op == "select":
    if isinstance(args[0], Const):
      return retype(args[1] if args[0].value else args[2], type)
  elif op == "not" and isinstance(args[0], Prim) and args[0].op == "not":
    return retype(args[0].args[0], type)
  return Prim(op, args, type)


def retype(e: Expr, type: ScalarType) -> Expr:
  """Reinterpret ``e`` as ``type`` when only the weak flag differs."""
  if e.type is type:
    return e
  assert e.type.name == type.name, (e.type, type)  # type: ignore[union-attr]
  if isinstance(e, Const):
    return Const(e.value, type)
  return Cast(e, type)


_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1


def cast(e: Expr, type: ScalarType) -> Expr:
  """Convert ``e`` to ``type`` with C/NumPy cast semantics."""
  src = e.type
  assert isinstance(src, ScalarType), src
  if src is type:
    return e
  if src.name == type.name:
    return retype(e, type)
  if isinstance(e, Const):
    v = e.value
    if type.kind == "b":
      return Const(v != 0, type)
    if type.kind == "f":
      return Const(type.dtype.type(v).item(), type)
    if src.kind == "f":
      if not math.isfinite(v) or not _I64_MIN <= v <= _I64_MAX:
        return Cast(e, type)
      v = int(v)
    return Const(np.array(int(v)).astype(type.dtype).item(), type)
  return Cast(e, type)


# ---- integer (index) arithmetic shorthands ---------------------------------


def _i(x) -> Expr:
  return Const(x, I) if isinstance(x, int) else x


def iadd(a, b) -> Expr:
  return prim("add", (_i(a), _i(b)), I)


def isub(a, b) -> Expr:
  return prim("sub", (_i(a), _i(b)), I)


def imul(a, b) -> Expr:
  return prim("mul", (_i(a), _i(b)), I)


def idiv(a, b) -> Expr:
  """Floor division of index values (callers guarantee ``b != 0``)."""
  return prim("floordiv", (_i(a), _i(b)), I)


def imod(a, b) -> Expr:
  return prim("mod", (_i(a), _i(b)), I)


def imin(a, b) -> Expr:
  return prim("min", (_i(a), _i(b)), I)


def imax(a, b) -> Expr:
  return prim("max", (_i(a), _i(b)), I)


def cmp(op: str, a, b) -> Expr:
  return prim(op, (_i(a), _i(b)), Bool)


def land(a: Expr, b: Expr) -> Expr:
  return prim("and", (a, b), Bool)


def lor(a: Expr, b: Expr) -> Expr:
  return prim("or", (a, b), Bool)


def lnot(a: Expr) -> Expr:
  return prim("not", (a,), Bool)


def select(c: Expr, a: Expr, b: Expr) -> Expr:
  return prim("select", (c, a, b), a.type)  # type: ignore[arg-type]


def iprod(xs) -> Expr:
  out: Expr = ONE
  for x in xs:
    out = imul(out, x)
  return out
