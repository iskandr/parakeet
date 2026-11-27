"""Direct PTX 7.8 emitter, dead-register tidy pass, and offline static verifier.

Bypasses NVRTC for plain ``ir.ParLoop`` kernels (maps, broadcasts, transposes,
gathers, Adam updates, atomic split-reductions) and ``WarpRowReducePlan``
kernels (coalesced warp-shuffle row reductions such as LayerNorm).  Kernels
that use unsupported operations (e.g. libm ``exp``/``log``/``tanh``/``erf``,
8/16-bit integer arithmetic, stack buffers, or tiled/monoidal shared-memory
schedules) raise :class:`PtxUnsupported` and return ``None`` so ``CuPyDriver``
routes them cleanly through NVRTC.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field

from ..ndtypes import BufType, ScalarType
from ..syntax import ir
from .cfamily import defined_vars
from .device import KernelPlan

# ---------------------------------------------------------------------------
# Types, errors, and structured PTX instruction representation
# ---------------------------------------------------------------------------


class PtxUnsupported(Exception):
  """Raised when an IR construct is outside the direct-PTX subset."""


class PtxCheckError(Exception):
  """Raised by :func:`verify_ptx` when generated PTX fails static checks."""


_SCALAR_REG_CLASS: dict[str, tuple[str, str, int]] = {
  # parakeet scalar name -> (reg_decl_type, reg_prefix, byte_size)
  "bool": (".pred", "%p", 1),
  "int32": (".b32", "%r", 4),
  "uint32": (".b32", "%r", 4),
  "int64": (".b64", "%rd", 8),
  "uint64": (".b64", "%rd", 8),
  "float32": (".f32", "%f", 4),
  "float64": (".f64", "%fd", 8),
}

_PTX_ARITH_TYPE: dict[str, str] = {
  "bool": "pred",
  "int32": "s32",
  "uint32": "u32",
  "int64": "s64",
  "uint64": "u64",
  "float32": "f32",
  "float64": "f64",
}

_PTX_PARAM_TYPE: dict[str, str] = {
  "bool": ".u8",
  "int32": ".s32",
  "uint32": ".u32",
  "int64": ".s64",
  "uint64": ".u64",
  "float32": ".f32",
  "float64": ".f64",
}

_BIT_TYPE: dict[str, str] = {
  "int32": "b32",
  "uint32": "b32",
  "int64": "b64",
  "uint64": "b64",
}


def _f32_hex(val: float) -> str:
  bits = struct.unpack(">I", struct.pack(">f", float(val)))[0]
  return f"0f{bits:08X}"


def _f64_hex(val: float) -> str:
  bits = struct.unpack(">Q", struct.pack(">d", float(val)))[0]
  return f"0d{bits:016X}"


def _const_literal(val, t: ScalarType) -> str:
  if t.name == "float32":
    return _f32_hex(float(val))
  if t.name == "float64":
    return _f64_hex(float(val))
  if t.name == "int64":
    return str(int(val))
  if t.name == "uint64":
    return f"{int(val) & 0xFFFFFFFFFFFFFFFF}U"
  if t.name == "uint32":
    return f"{int(val) & 0xFFFFFFFF}U"
  if t.name == "int32":
    return str(int(val))
  raise PtxUnsupported(f"unsupported constant type {t.name}")


@dataclass
class PtxInst:
  """One emitted PTX instruction or label with explicit def/use register sets."""

  text: str
  defs: tuple[str, ...] = ()
  uses: tuple[str, ...] = ()
  side_effect: bool = False
  is_label: bool = False


@dataclass
class PtxEntry:
  """Structured PTX kernel entry before tidy and serialization."""

  name: str
  params: list[tuple[str, str]]  # (ptx_param_type, param_name)
  reg_types: dict[str, str] = field(default_factory=dict)  # reg_name -> .pred/.b16/.b32/.b64/.f32/.f64
  insts: list[PtxInst] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Virtual-register SSA PTX emitter
# ---------------------------------------------------------------------------


class PtxEmitter:
  """Lowers Parakeet loop IR into virtual-register PTX 7.8 instructions."""

  def __init__(self) -> None:
    self._counters: dict[str, int] = {"%p": 0, "%h": 0, "%r": 0, "%rd": 0, "%f": 0, "%fd": 0}
    self._label_id = 0
    self.reg_types: dict[str, str] = {}
    self.insts: list[PtxInst] = []
    self.var_regs: dict[int, str] = {}
    self.buf_regs: dict[int, str] = {}
    self.errors: list[ir.ErrorInfo] = []
    self._error_ids: dict[ir.ErrorInfo, int] = {}
    self._loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)
    self.flag_reg: str = ""
    self.info_reg: str = ""

  # ---- register and label allocation ---------------------------------------

  def _new_reg_raw(self, decl_type: str, prefix: str) -> str:
    idx = self._counters[prefix]
    self._counters[prefix] = idx + 1
    name = f"{prefix}{idx}"
    self.reg_types[name] = decl_type
    return name

  def new_reg(self, t: ScalarType) -> str:
    info = _SCALAR_REG_CLASS.get(t.name)
    if info is None:
      raise PtxUnsupported(f"scalar type {t.name} not supported in direct PTX")
    decl_type, prefix, _ = info
    return self._new_reg_raw(decl_type, prefix)

  def new_pred(self) -> str:
    return self._new_reg_raw(".pred", "%p")

  def new_b16(self) -> str:
    return self._new_reg_raw(".b16", "%h")

  def new_b32(self) -> str:
    return self._new_reg_raw(".b32", "%r")

  def new_b64(self) -> str:
    return self._new_reg_raw(".b64", "%rd")

  def new_f32(self) -> str:
    return self._new_reg_raw(".f32", "%f")

  def new_f64(self) -> str:
    return self._new_reg_raw(".f64", "%fd")

  def new_label(self, prefix: str = "L") -> str:
    idx = self._label_id
    self._label_id += 1
    return f"{prefix}_{idx}"

  def emit(
    self,
    text: str,
    *,
    defs: tuple[str, ...] = (),
    uses: tuple[str, ...] = (),
    side_effect: bool = False,
  ) -> None:
    self.insts.append(PtxInst(text=text, defs=defs, uses=uses, side_effect=side_effect))

  def emit_label(self, label: str) -> None:
    self.insts.append(PtxInst(text=f"{label}:", side_effect=True, is_label=True))

  def error_id(self, err: ir.ErrorInfo) -> int:
    eid = self._error_ids.get(err)
    if eid is None:
      eid = len(self.errors)
      self.errors.append(err)
      self._error_ids[err] = eid
    return eid

  # ---- helpers -------------------------------------------------------------

  def emit_zero_reg(self, reg: str, t: ScalarType) -> None:
    if t.name == "bool":
      z = self.new_b32()
      self.emit(f"mov.u32 {z}, 0;", defs=(z,))
      self.emit(f"setp.ne.u32 {reg}, {z}, 0;", defs=(reg,), uses=(z,))
    elif t.name == "float32":
      self.emit(f"mov.f32 {reg}, {_f32_hex(0.0)};", defs=(reg,))
    elif t.name == "float64":
      self.emit(f"mov.f64 {reg}, {_f64_hex(0.0)};", defs=(reg,))
    elif t.name in ("int32", "uint32"):
      self.emit(f"mov.b32 {reg}, 0;", defs=(reg,))
    elif t.name in ("int64", "uint64"):
      self.emit(f"mov.b64 {reg}, 0;", defs=(reg,))
    else:
      raise PtxUnsupported(f"unsupported zero init for {t.name}")

  def emit_const_into(self, reg: str, val, t: ScalarType) -> None:
    if t.name == "bool":
      z = self.new_b32()
      self.emit(f"mov.u32 {z}, 0;", defs=(z,))
      if bool(val):
        self.emit(f"setp.eq.u32 {reg}, {z}, 0;", defs=(reg,), uses=(z,))
      else:
        self.emit(f"setp.ne.u32 {reg}, {z}, 0;", defs=(reg,), uses=(z,))
      return
    lit = _const_literal(val, t)
    mov_ty = _PTX_ARITH_TYPE[t.name]
    self.emit(f"mov.{mov_ty} {reg}, {lit};", defs=(reg,))

  def emit_mov(self, dst: str, src: str, t: ScalarType) -> None:
    if dst == src:
      return
    if t.name == "bool":
      self.emit(f"mov.pred {dst}, {src};", defs=(dst,), uses=(src,))
    elif t.name in ("int32", "uint32"):
      self.emit(f"mov.b32 {dst}, {src};", defs=(dst,), uses=(src,))
    elif t.name in ("int64", "uint64"):
      self.emit(f"mov.b64 {dst}, {src};", defs=(dst,), uses=(src,))
    elif t.name == "float32":
      self.emit(f"mov.f32 {dst}, {src};", defs=(dst,), uses=(src,))
    elif t.name == "float64":
      self.emit(f"mov.f64 {dst}, {src};", defs=(dst,), uses=(src,))
    else:
      raise PtxUnsupported(f"unsupported mov for {t.name}")

  def ensure_var_reg(self, v: ir.Var) -> str:
    reg = self.var_regs.get(v.uid)
    if reg is None:
      reg = self.new_reg(v.type)
      self.var_regs[v.uid] = reg
    return reg

  def _to_i64(self, reg: str, t: ScalarType) -> str:
    if t.name == "int64":
      return reg
    if t.name == "uint64":
      return reg
    dst = self.new_b64()
    if t.name == "int32":
      self.emit(f"cvt.s64.s32 {dst}, {reg};", defs=(dst,), uses=(reg,))
    elif t.name == "uint32":
      self.emit(f"cvt.u64.u32 {dst}, {reg};", defs=(dst,), uses=(reg,))
    elif t.name == "bool":
      self.emit(f"selp.s64 {dst}, 1, 0, {reg};", defs=(dst,), uses=(reg,))
    else:
      raise PtxUnsupported(f"cannot widen {t.name} to 64-bit index")
    return dst

  def _elem_addr(self, buf: ir.Var, idx_expr: ir.Expr) -> str:
    base = self.buf_regs.get(buf.uid)
    if base is None:
      raise PtxUnsupported(f"unmapped buffer {buf.name}")
    idx_r = self._to_i64(self.expr(idx_expr), idx_expr.type)
    elt = buf.type.elt
    info = _SCALAR_REG_CLASS.get(elt.name)
    if info is None:
      raise PtxUnsupported(f"unsupported element type {elt.name}")
    itemsize = info[2]
    if itemsize == 1:
      off = idx_r
    else:
      shift = itemsize.bit_length() - 1
      off = self.new_b64()
      self.emit(f"shl.b64 {off}, {idx_r}, {shift};", defs=(off,), uses=(idx_r,))
    addr = self.new_b64()
    self.emit(f"add.u64 {addr}, {base}, {off};", defs=(addr,), uses=(base, off))
    return addr

  # ---- expressions ---------------------------------------------------------

  def expr(self, e: ir.Expr) -> str:
    if isinstance(e, ir.Const):
      r = self.new_reg(e.type)
      self.emit_const_into(r, e.value, e.type)
      return r
    if isinstance(e, ir.Var):
      if isinstance(e.type, BufType):
        raise PtxUnsupported("buffer value used as scalar expression")
      reg = self.var_regs.get(e.uid)
      if reg is None:
        raise PtxUnsupported(f"undefined scalar var {e.name}")
      return reg
    if isinstance(e, ir.Load):
      return self._emit_load(e)
    if isinstance(e, ir.Cast):
      return self._emit_cast(e)
    if isinstance(e, ir.Prim):
      return self._emit_prim(e)
    raise PtxUnsupported(f"unsupported expression {type(e).__name__}")

  def _emit_load(self, e: ir.Load) -> str:
    addr = self._elem_addr(e.buf, e.index)
    elt = e.buf.type.elt
    if elt.name == "bool":
      tmp = self.new_b16()
      dst = self.new_pred()
      self.emit(f"ld.global.u8 {tmp}, [{addr}];", defs=(tmp,), uses=(addr,))
      self.emit(f"setp.ne.u16 {dst}, {tmp}, 0;", defs=(dst,), uses=(tmp,))
      return dst
    ptx_ty = _PTX_ARITH_TYPE.get(elt.name)
    if ptx_ty is None:
      raise PtxUnsupported(f"unsupported load type {elt.name}")
    dst = self.new_reg(elt)
    self.emit(f"ld.global.{ptx_ty} {dst}, [{addr}];", defs=(dst,), uses=(addr,))
    return dst

  def _emit_cast(self, e: ir.Cast) -> str:
    src_t, dst_t = e.arg.type, e.type
    x = self.expr(e.arg)
    if src_t.name == dst_t.name:
      return x
    if src_t.name not in _SCALAR_REG_CLASS or dst_t.name not in _SCALAR_REG_CLASS:
      raise PtxUnsupported(f"cast {src_t.name} -> {dst_t.name} unsupported")

    # -> bool
    if dst_t.name == "bool":
      dst = self.new_pred()
      if src_t.name in ("int32", "uint32"):
        self.emit(f"setp.ne.u32 {dst}, {x}, 0;", defs=(dst,), uses=(x,))
      elif src_t.name in ("int64", "uint64"):
        self.emit(f"setp.ne.u64 {dst}, {x}, 0;", defs=(dst,), uses=(x,))
      elif src_t.name == "float32":
        self.emit(f"setp.neu.f32 {dst}, {x}, {_f32_hex(0.0)};", defs=(dst,), uses=(x,))
      elif src_t.name == "float64":
        self.emit(f"setp.neu.f64 {dst}, {x}, {_f64_hex(0.0)};", defs=(dst,), uses=(x,))
      return dst

    # bool -> numeric
    if src_t.name == "bool":
      dst = self.new_reg(dst_t)
      one = _const_literal(1, dst_t)
      zero = _const_literal(0, dst_t)
      ty = _PTX_ARITH_TYPE[dst_t.name]
      self.emit(f"selp.{ty} {dst}, {one}, {zero}, {x};", defs=(dst,), uses=(x,))
      return dst

    # float -> int (keep NVRTC fallback for exact pk_f2i64 / pk_f2u64 edge semantics)
    if src_t.kind == "f" and dst_t.kind in "iu":
      raise PtxUnsupported(f"float-to-int cast {src_t.name} -> {dst_t.name} uses NVRTC helper")

    dst = self.new_reg(dst_t)
    s_ty = _PTX_ARITH_TYPE[src_t.name]
    d_ty = _PTX_ARITH_TYPE[dst_t.name]

    # Same-width integer reinterpret
    if src_t.kind in "iu" and dst_t.kind in "iu" and src_t.itemsize == dst_t.itemsize:
      bty = "b32" if dst_t.itemsize == 4 else "b64"
      self.emit(f"mov.{bty} {dst}, {x};", defs=(dst,), uses=(x,))
      return dst

    # int -> int widening/narrowing
    if src_t.kind in "iu" and dst_t.kind in "iu":
      self.emit(f"cvt.{d_ty}.{s_ty} {dst}, {x};", defs=(dst,), uses=(x,))
      return dst

    # int -> float
    if src_t.kind in "iu" and dst_t.kind == "f":
      self.emit(f"cvt.rn.{d_ty}.{s_ty} {dst}, {x};", defs=(dst,), uses=(x,))
      return dst

    # float <-> float
    if src_t.name == "float32" and dst_t.name == "float64":
      self.emit(f"cvt.f64.f32 {dst}, {x};", defs=(dst,), uses=(x,))
      return dst
    if src_t.name == "float64" and dst_t.name == "float32":
      self.emit(f"cvt.rn.f32.f64 {dst}, {x};", defs=(dst,), uses=(x,))
      return dst

    raise PtxUnsupported(f"unsupported cast {src_t.name} -> {dst_t.name}")

  def _emit_prim(self, e: ir.Prim) -> str:
    op, t = e.op, e.type
    at = e.args[0].type
    if t.name not in _SCALAR_REG_CLASS or any(a.type.name not in _SCALAR_REG_CLASS for a in e.args):
      raise PtxUnsupported(f"unsupported types for prim {op}")

    # Logical / predicate ops
    if op == "and":
      a, b = self.expr(e.args[0]), self.expr(e.args[1])
      dst = self.new_pred()
      self.emit(f"and.pred {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst
    if op == "or":
      a, b = self.expr(e.args[0]), self.expr(e.args[1])
      dst = self.new_pred()
      self.emit(f"or.pred {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst
    if op == "not":
      a = self.expr(e.args[0])
      dst = self.new_pred()
      self.emit(f"not.pred {dst}, {a};", defs=(dst,), uses=(a,))
      return dst

    # Comparisons
    if op in ("eq", "ne", "lt", "le", "gt", "ge"):
      a, b = self.expr(e.args[0]), self.expr(e.args[1])
      dst = self.new_pred()
      if at.name == "bool":
        if op == "ne":
          self.emit(f"xor.pred {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
        elif op == "eq":
          tmp = self.new_pred()
          self.emit(f"xor.pred {tmp}, {a}, {b};", defs=(tmp,), uses=(a, b))
          self.emit(f"not.pred {dst}, {tmp};", defs=(dst,), uses=(tmp,))
        else:
          raise PtxUnsupported(f"ordering comparison {op} on bool")
        return dst
      cmp_op = "neu" if (op == "ne" and at.kind == "f") else op
      aty = _PTX_ARITH_TYPE[at.name]
      self.emit(f"setp.{cmp_op}.{aty} {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst

    # Select
    if op == "select":
      cond = self.expr(e.args[0])
      vt = self.expr(e.args[1])
      vf = self.expr(e.args[2])
      if t.name == "bool":
        # (cond & vt) | (!cond & vf)
        t1, nc, t2, dst = self.new_pred(), self.new_pred(), self.new_pred(), self.new_pred()
        self.emit(f"and.pred {t1}, {cond}, {vt};", defs=(t1,), uses=(cond, vt))
        self.emit(f"not.pred {nc}, {cond};", defs=(nc,), uses=(cond,))
        self.emit(f"and.pred {t2}, {nc}, {vf};", defs=(t2,), uses=(nc, vf))
        self.emit(f"or.pred {dst}, {t1}, {t2};", defs=(dst,), uses=(t1, t2))
        return dst
      dst = self.new_reg(t)
      ty = _PTX_ARITH_TYPE[t.name]
      self.emit(f"selp.{ty} {dst}, {vt}, {vf}, {cond};", defs=(dst,), uses=(vt, vf, cond))
      return dst

    # Float classification predicates
    if op == "isnan":
      a = self.expr(e.args[0])
      dst = self.new_pred()
      aty = _PTX_ARITH_TYPE[at.name]
      self.emit(f"setp.neu.{aty} {dst}, {a}, {a};", defs=(dst,), uses=(a,))
      return dst
    if op in ("isinf", "isfinite"):
      a = self.expr(e.args[0])
      abs_r = self.new_reg(at)
      dst = self.new_pred()
      aty = _PTX_ARITH_TYPE[at.name]
      inf_lit = _f32_hex(math.inf) if at.name == "float32" else _f64_hex(math.inf)
      cmp_op = "eq" if op == "isinf" else "lt"
      self.emit(f"abs.{aty} {abs_r}, {a};", defs=(abs_r,), uses=(a,))
      self.emit(f"setp.{cmp_op}.{aty} {dst}, {abs_r}, {inf_lit};", defs=(dst,), uses=(abs_r,))
      return dst
    if op == "signbit":
      a = self.expr(e.args[0])
      dst = self.new_pred()
      if at.name == "float32":
        bits = self.new_b32()
        self.emit(f"mov.b32 {bits}, {a};", defs=(bits,), uses=(a,))
        self.emit(f"setp.lt.s32 {dst}, {bits}, 0;", defs=(dst,), uses=(bits,))
      else:
        bits = self.new_b64()
        self.emit(f"mov.b64 {bits}, {a};", defs=(bits,), uses=(a,))
        self.emit(f"setp.lt.s64 {dst}, {bits}, 0;", defs=(dst,), uses=(bits,))
      return dst

    if t.kind == "f":
      return self._emit_float_prim(op, e.args, t)
    if t.kind in "iu":
      return self._emit_int_prim(op, e.args, t)
    raise PtxUnsupported(f"unsupported prim {op} on {t.name}")

  def _emit_float_prim(self, op: str, args: tuple[ir.Expr, ...], t: ScalarType) -> str:
    ty = _PTX_ARITH_TYPE[t.name]
    if op in ("add", "sub", "mul"):
      a, b = self.expr(args[0]), self.expr(args[1])
      dst = self.new_reg(t)
      self.emit(f"{op}.{ty} {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst
    if op == "div":
      a, b = self.expr(args[0]), self.expr(args[1])
      dst = self.new_reg(t)
      self.emit(f"div.rn.{ty} {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst
    if op == "neg":
      a = self.expr(args[0])
      dst = self.new_reg(t)
      self.emit(f"neg.{ty} {dst}, {a};", defs=(dst,), uses=(a,))
      return dst
    if op == "abs":
      a = self.expr(args[0])
      dst = self.new_reg(t)
      self.emit(f"abs.{ty} {dst}, {a};", defs=(dst,), uses=(a,))
      return dst
    if op == "sqrt":
      a = self.expr(args[0])
      dst = self.new_reg(t)
      self.emit(f"sqrt.rn.{ty} {dst}, {a};", defs=(dst,), uses=(a,))
      return dst
    if op in ("min", "max"):
      # Exact NaN-propagating semantics matching pk_min_float32/pk_max_float32:
      # (a != a) ? a : ((b != b) ? b : (a < b ? a : b))
      a, b = self.expr(args[0]), self.expr(args[1])
      p_an, p_bn, p_cmp = self.new_pred(), self.new_pred(), self.new_pred()
      t1, t2, dst = self.new_reg(t), self.new_reg(t), self.new_reg(t)
      cmp_op = "lt" if op == "min" else "gt"
      self.emit(f"setp.{cmp_op}.{ty} {p_cmp}, {a}, {b};", defs=(p_cmp,), uses=(a, b))
      self.emit(f"selp.{ty} {t1}, {a}, {b}, {p_cmp};", defs=(t1,), uses=(a, b, p_cmp))
      self.emit(f"setp.neu.{ty} {p_bn}, {b}, {b};", defs=(p_bn,), uses=(b,))
      self.emit(f"selp.{ty} {t2}, {b}, {t1}, {p_bn};", defs=(t2,), uses=(b, t1, p_bn))
      self.emit(f"setp.neu.{ty} {p_an}, {a}, {a};", defs=(p_an,), uses=(a,))
      self.emit(f"selp.{ty} {dst}, {a}, {t2}, {p_an};", defs=(dst,), uses=(a, t2, p_an))
      return dst
    raise PtxUnsupported(f"float op {op} falls back to NVRTC")

  def _emit_int_prim(self, op: str, args: tuple[ir.Expr, ...], t: ScalarType) -> str:
    ty = _PTX_ARITH_TYPE[t.name]
    bty = _BIT_TYPE[t.name]
    signed = t.kind == "i"
    bits = t.itemsize * 8

    if op in ("add", "sub", "min", "max"):
      a, b = self.expr(args[0]), self.expr(args[1])
      dst = self.new_reg(t)
      self.emit(f"{op}.{ty} {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst
    if op == "mul":
      a, b = self.expr(args[0]), self.expr(args[1])
      dst = self.new_reg(t)
      self.emit(f"mul.lo.{ty} {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst
    if op == "neg":
      a = self.expr(args[0])
      dst = self.new_reg(t)
      if signed:
        self.emit(f"neg.{ty} {dst}, {a};", defs=(dst,), uses=(a,))
      else:
        zero = self.new_reg(t)
        self.emit(f"mov.{bty} {zero}, 0;", defs=(zero,))
        self.emit(f"sub.{ty} {dst}, {zero}, {a};", defs=(dst,), uses=(zero, a))
      return dst
    if op == "abs":
      a = self.expr(args[0])
      if not signed:
        return a
      dst = self.new_reg(t)
      self.emit(f"abs.{ty} {dst}, {a};", defs=(dst,), uses=(a,))
      return dst
    if op in ("bitand", "bitor", "bitxor"):
      a, b = self.expr(args[0]), self.expr(args[1])
      dst = self.new_reg(t)
      ptx_op = {"bitand": "and", "bitor": "or", "bitxor": "xor"}[op]
      self.emit(f"{ptx_op}.{bty} {dst}, {a}, {b};", defs=(dst,), uses=(a, b))
      return dst
    if op == "invert":
      a = self.expr(args[0])
      dst = self.new_reg(t)
      self.emit(f"not.{bty} {dst}, {a};", defs=(dst,), uses=(a,))
      return dst
    if op in ("lshift", "rshift"):
      a, b = self.expr(args[0]), self.expr(args[1])
      # Match pk_shl / pk_shr: out-of-range shift returns 0 (or -1 for signed negative rshift)
      p_oob = self.new_pred()
      if signed:
        p_neg, p_hi = self.new_pred(), self.new_pred()
        self.emit(f"setp.lt.{ty} {p_neg}, {b}, 0;", defs=(p_neg,), uses=(b,))
        self.emit(f"setp.ge.{ty} {p_hi}, {b}, {bits};", defs=(p_hi,), uses=(b,))
        self.emit(f"or.pred {p_oob}, {p_neg}, {p_hi};", defs=(p_oob,), uses=(p_neg, p_hi))
      else:
        self.emit(f"setp.ge.{ty} {p_oob}, {b}, {bits};", defs=(p_oob,), uses=(b,))
      # PTX shift amount must be 32-bit unsigned
      if bits == 64:
        b32 = self.new_b32()
        self.emit(f"cvt.u32.u64 {b32}, {b};", defs=(b32,), uses=(b,))
      else:
        b32 = b
      safe_b32 = self.new_b32()
      self.emit(f"selp.u32 {safe_b32}, 0, {b32}, {p_oob};", defs=(safe_b32,), uses=(b32, p_oob))
      shifted = self.new_reg(t)
      if op == "lshift":
        self.emit(f"shl.{bty} {shifted}, {a}, {safe_b32};", defs=(shifted,), uses=(a, safe_b32))
        fallback = self.new_reg(t)
        self.emit(f"mov.{bty} {fallback}, 0;", defs=(fallback,))
      else:
        self.emit(f"shr.{ty} {shifted}, {a}, {safe_b32};", defs=(shifted,), uses=(a, safe_b32))
        if signed:
          p_aneg = self.new_pred()
          fallback = self.new_reg(t)
          self.emit(f"setp.lt.{ty} {p_aneg}, {a}, 0;", defs=(p_aneg,), uses=(a,))
          self.emit(f"selp.{ty} {fallback}, -1, 0, {p_aneg};", defs=(fallback,), uses=(p_aneg,))
        else:
          fallback = self.new_reg(t)
          self.emit(f"mov.{bty} {fallback}, 0;", defs=(fallback,))
      dst = self.new_reg(t)
      self.emit(f"selp.{ty} {dst}, {fallback}, {shifted}, {p_oob};", defs=(dst,), uses=(fallback, shifted, p_oob))
      return dst
    if op in ("floordiv", "mod", "fmod"):
      return self._emit_int_divmod(op, args, t)
    if op == "popcount":
      a = self.expr(args[0])
      at = args[0].type
      dst = self.new_reg(t)
      pop32 = self.new_b32()
      if at.itemsize == 4:
        self.emit(f"popc.b32 {pop32}, {a};", defs=(pop32,), uses=(a,))
      elif at.itemsize == 8:
        self.emit(f"popc.b64 {pop32}, {a};", defs=(pop32,), uses=(a,))
      else:
        raise PtxUnsupported(f"popcount on {at.name}")
      if t.itemsize == 4:
        self.emit(f"mov.b32 {dst}, {pop32};", defs=(dst,), uses=(pop32,))
      else:
        self.emit(f"cvt.{ty}.u32 {dst}, {pop32};", defs=(dst,), uses=(pop32,))
      return dst

    raise PtxUnsupported(f"integer op {op} falls back to NVRTC")

  def _emit_int_divmod(self, op: str, args: tuple[ir.Expr, ...], t: ScalarType) -> str:
    a, b = self.expr(args[0]), self.expr(args[1])
    ty = _PTX_ARITH_TYPE[t.name]
    bty = _BIT_TYPE[t.name]
    signed = t.kind == "i"

    p_zero = self.new_pred()
    self.emit(f"setp.eq.{ty} {p_zero}, {b}, 0;", defs=(p_zero,), uses=(b,))

    if not signed:
      safe_b = self.new_reg(t)
      self.emit(f"selp.{ty} {safe_b}, 1, {b}, {p_zero};", defs=(safe_b,), uses=(b, p_zero))
      raw = self.new_reg(t)
      ptx_op = "div" if op == "floordiv" else "rem"
      self.emit(f"{ptx_op}.{ty} {raw}, {a}, {safe_b};", defs=(raw,), uses=(a, safe_b))
      dst = self.new_reg(t)
      self.emit(f"selp.{ty} {dst}, 0, {raw}, {p_zero};", defs=(dst,), uses=(raw, p_zero))
      return dst

    p_neg1 = self.new_pred()
    p_special = self.new_pred()
    self.emit(f"setp.eq.{ty} {p_neg1}, {b}, -1;", defs=(p_neg1,), uses=(b,))
    self.emit(f"or.pred {p_special}, {p_zero}, {p_neg1};", defs=(p_special,), uses=(p_zero, p_neg1))
    safe_b = self.new_reg(t)
    self.emit(f"selp.{ty} {safe_b}, 1, {b}, {p_special};", defs=(safe_b,), uses=(b, p_special))

    if op == "fmod":
      r = self.new_reg(t)
      self.emit(f"rem.{ty} {r}, {a}, {safe_b};", defs=(r,), uses=(a, safe_b))
      dst = self.new_reg(t)
      self.emit(f"selp.{ty} {dst}, 0, {r}, {p_special};", defs=(dst,), uses=(r, p_special))
      return dst

    r = self.new_reg(t)
    self.emit(f"rem.{ty} {r}, {a}, {safe_b};", defs=(r,), uses=(a, safe_b))
    p_rnz = self.new_pred()
    self.emit(f"setp.ne.{ty} {p_rnz}, {r}, 0;", defs=(p_rnz,), uses=(r,))

    if op == "floordiv":
      q = self.new_reg(t)
      self.emit(f"div.{ty} {q}, {a}, {safe_b};", defs=(q,), uses=(a, safe_b))
      xab = self.new_reg(t)
      p_diff, p_adj = self.new_pred(), self.new_pred()
      q_m1, q_adj = self.new_reg(t), self.new_reg(t)
      neg_a, q_n1, dst = self.new_reg(t), self.new_reg(t), self.new_reg(t)
      self.emit(f"xor.{bty} {xab}, {a}, {b};", defs=(xab,), uses=(a, b))
      self.emit(f"setp.lt.{ty} {p_diff}, {xab}, 0;", defs=(p_diff,), uses=(xab,))
      self.emit(f"and.pred {p_adj}, {p_rnz}, {p_diff};", defs=(p_adj,), uses=(p_rnz, p_diff))
      self.emit(f"sub.{ty} {q_m1}, {q}, 1;", defs=(q_m1,), uses=(q,))
      self.emit(f"selp.{ty} {q_adj}, {q_m1}, {q}, {p_adj};", defs=(q_adj,), uses=(q_m1, q, p_adj))
      self.emit(f"neg.{ty} {neg_a}, {a};", defs=(neg_a,), uses=(a,))
      self.emit(f"selp.{ty} {q_n1}, {neg_a}, {q_adj}, {p_neg1};", defs=(q_n1,), uses=(neg_a, q_adj, p_neg1))
      self.emit(f"selp.{ty} {dst}, 0, {q_n1}, {p_zero};", defs=(dst,), uses=(q_n1, p_zero))
      return dst

    # Python mod: r != 0 && ((r < 0) != (b < 0)) -> r + b
    xrb = self.new_reg(t)
    p_diff, p_adj = self.new_pred(), self.new_pred()
    r_pb, r_adj, dst = self.new_reg(t), self.new_reg(t), self.new_reg(t)
    self.emit(f"xor.{bty} {xrb}, {r}, {b};", defs=(xrb,), uses=(r, b))
    self.emit(f"setp.lt.{ty} {p_diff}, {xrb}, 0;", defs=(p_diff,), uses=(xrb,))
    self.emit(f"and.pred {p_adj}, {p_rnz}, {p_diff};", defs=(p_adj,), uses=(p_rnz, p_diff))
    self.emit(f"add.{ty} {r_pb}, {r}, {b};", defs=(r_pb,), uses=(r, b))
    self.emit(f"selp.{ty} {r_adj}, {r_pb}, {r}, {p_adj};", defs=(r_adj,), uses=(r_pb, r, p_adj))
    self.emit(f"selp.{ty} {dst}, 0, {r_adj}, {p_special};", defs=(dst,), uses=(r_adj, p_special))
    return dst

  # ---- statements ----------------------------------------------------------

  def block(self, stmts: list[ir.Stmt]) -> None:
    for s in stmts:
      self.stmt(s)

  def stmt(self, s: ir.Stmt) -> None:
    if isinstance(s, ir.Assign):
      if isinstance(s.var.type, BufType):
        raise PtxUnsupported("buffer assignment inside kernel")
      val_r = self.expr(s.value)
      dst_r = self.ensure_var_reg(s.var)
      self.emit_mov(dst_r, val_r, s.var.type)
    elif isinstance(s, ir.Store):
      self._emit_store(s)
    elif isinstance(s, ir.If):
      self._emit_if(s)
    elif isinstance(s, ir.ForRange):
      self._emit_for_range(s)
    elif isinstance(s, ir.While):
      self._emit_while(s)
    elif isinstance(s, ir.Break):
      if not self._loop_stack:
        raise PtxUnsupported("break outside loop")
      self.emit(f"bra {self._loop_stack[-1][1]};", side_effect=True)
    elif isinstance(s, ir.Continue):
      if not self._loop_stack:
        raise PtxUnsupported("continue outside loop")
      self.emit(f"bra {self._loop_stack[-1][0]};", side_effect=True)
    elif isinstance(s, ir.Check):
      self._emit_check(s)
    elif isinstance(s, ir.Comment):
      pass
    elif isinstance(s, ir.ReleaseScope):
      if s.scope.owned:
        raise PtxUnsupported("non-empty ReleaseScope inside kernel")
    else:
      raise PtxUnsupported(f"unsupported statement {type(s).__name__}")

  def _emit_store(self, s: ir.Store) -> None:
    addr = self._elem_addr(s.buf, s.index)
    val = self.expr(s.value)
    elt = s.buf.type.elt
    if s.atomic:
      if s.atomic == "add" and elt.name in ("float32", "float64", "int32", "uint32"):
        ptx_ty = _PTX_ARITH_TYPE[elt.name]
        old = self.new_reg(elt)
        self.emit(f"atom.global.add.{ptx_ty} {old}, [{addr}], {val};", defs=(old,), uses=(addr, val), side_effect=True)
        return
      raise PtxUnsupported(f"unsupported atomic store {s.atomic} on {elt.name}")
    if elt.name == "bool":
      tmp = self.new_b16()
      self.emit(f"selp.u16 {tmp}, 1, 0, {val};", defs=(tmp,), uses=(val,))
      self.emit(f"st.global.u8 [{addr}], {tmp};", uses=(addr, tmp), side_effect=True)
      return
    ptx_ty = _PTX_ARITH_TYPE.get(elt.name)
    if ptx_ty is None:
      raise PtxUnsupported(f"unsupported store type {elt.name}")
    self.emit(f"st.global.{ptx_ty} [{addr}], {val};", uses=(addr, val), side_effect=True)

  def _emit_if(self, s: ir.If) -> None:
    cond = self.expr(s.cond)
    if s.orelse:
      l_else = self.new_label("L_else")
      l_end = self.new_label("L_endif")
      self.emit(f"@!{cond} bra {l_else};", uses=(cond,), side_effect=True)
      self.block(s.then)
      self.emit(f"bra {l_end};", side_effect=True)
      self.emit_label(l_else)
      self.block(s.orelse)
      self.emit_label(l_end)
    else:
      l_end = self.new_label("L_endif")
      self.emit(f"@!{cond} bra {l_end};", uses=(cond,), side_effect=True)
      self.block(s.then)
      self.emit_label(l_end)

  def _emit_for_range(self, s: ir.ForRange) -> None:
    iv_t = s.var.type
    if iv_t.name not in ("int32", "int64"):
      raise PtxUnsupported(f"ForRange index type {iv_t.name}")
    ty = _PTX_ARITH_TYPE[iv_t.name]
    iv_r = self.ensure_var_reg(s.var)
    start_r = self.expr(s.start)
    stop_r = self.expr(s.stop)
    step_r = self.expr(s.step) if s.step is not None else None
    self.emit_mov(iv_r, start_r, iv_t)

    l_head = self.new_label("L_for_head")
    l_cont = self.new_label("L_for_cont")
    l_end = self.new_label("L_for_end")

    self.emit_label(l_head)
    p_done = self.new_pred()
    if step_r is None or (isinstance(s.step, ir.Const) and int(s.step.value) > 0):
      self.emit(f"setp.ge.{ty} {p_done}, {iv_r}, {stop_r};", defs=(p_done,), uses=(iv_r, stop_r))
    elif isinstance(s.step, ir.Const) and int(s.step.value) < 0:
      self.emit(f"setp.le.{ty} {p_done}, {iv_r}, {stop_r};", defs=(p_done,), uses=(iv_r, stop_r))
    else:
      p_pos, p_ge, p_le = self.new_pred(), self.new_pred(), self.new_pred()
      p_t1, p_neg, p_t2 = self.new_pred(), self.new_pred(), self.new_pred()
      self.emit(f"setp.gt.{ty} {p_pos}, {step_r}, 0;", defs=(p_pos,), uses=(step_r,))
      self.emit(f"setp.ge.{ty} {p_ge}, {iv_r}, {stop_r};", defs=(p_ge,), uses=(iv_r, stop_r))
      self.emit(f"setp.le.{ty} {p_le}, {iv_r}, {stop_r};", defs=(p_le,), uses=(iv_r, stop_r))
      self.emit(f"and.pred {p_t1}, {p_pos}, {p_ge};", defs=(p_t1,), uses=(p_pos, p_ge))
      self.emit(f"not.pred {p_neg}, {p_pos};", defs=(p_neg,), uses=(p_pos,))
      self.emit(f"and.pred {p_t2}, {p_neg}, {p_le};", defs=(p_t2,), uses=(p_neg, p_le))
      self.emit(f"or.pred {p_done}, {p_t1}, {p_t2};", defs=(p_done,), uses=(p_t1, p_t2))
    self.emit(f"@{p_done} bra {l_end};", uses=(p_done,), side_effect=True)

    self._loop_stack.append((l_cont, l_end))
    self.block(s.body)
    self._loop_stack.pop()

    self.emit_label(l_cont)
    if step_r is None:
      self.emit(f"add.{ty} {iv_r}, {iv_r}, 1;", defs=(iv_r,), uses=(iv_r,))
    else:
      self.emit(f"add.{ty} {iv_r}, {iv_r}, {step_r};", defs=(iv_r,), uses=(iv_r, step_r))
    self.emit(f"bra {l_head};", side_effect=True)
    self.emit_label(l_end)

  def _emit_while(self, s: ir.While) -> None:
    l_head = self.new_label("L_while_head")
    l_end = self.new_label("L_while_end")
    self.emit_label(l_head)
    self._loop_stack.append((l_head, l_end))
    self.block(s.body)
    self._loop_stack.pop()
    self.emit(f"bra {l_head};", side_effect=True)
    self.emit_label(l_end)

  def _emit_check(self, s: ir.Check) -> None:
    cond = self.expr(s.cond)
    eid = self.error_id(s.err)
    l_ok = self.new_label("L_chk_ok")
    l_ret = self.new_label("L_chk_ret")
    self.emit(f"@{cond} bra {l_ok};", uses=(cond,), side_effect=True)
    self._emit_fail_body(eid, [self._to_i64(self.expr(a), a.type) for a in s.args[:4]], l_ret)
    self.emit_label(l_ok)

  def _emit_fail_body(self, eid: int, arg_regs: list[str], l_ret: str) -> None:
    old = self.new_b32()
    p_lost = self.new_pred()
    self.emit(
      f"atom.global.cas.b32 {old}, [{self.flag_reg}], 0, {eid + 1};",
      defs=(old,),
      uses=(self.flag_reg,),
      side_effect=True,
    )
    self.emit(f"setp.ne.u32 {p_lost}, {old}, 0;", defs=(p_lost,), uses=(old,))
    self.emit(f"@{p_lost} bra {l_ret};", uses=(p_lost,), side_effect=True)
    for slot in range(4):
      if slot < len(arg_regs):
        ar = arg_regs[slot]
      else:
        ar = self.new_b64()
        self.emit(f"mov.s64 {ar}, 0;", defs=(ar,))
      off_txt = f"+{slot * 8}" if slot > 0 else ""
      self.emit(f"st.global.s64 [{self.info_reg}{off_txt}], {ar};", uses=(self.info_reg, ar), side_effect=True)
    self.emit_label(l_ret)
    self.emit("ret;", side_effect=True)

  # ---- parameter setup & warp shuffle helpers ------------------------------

  def load_params(self, scalars: list[ir.Var], bufs: list[ir.Var]) -> tuple[list[tuple[str, str]], str, str]:
    params: list[tuple[str, str]] = [(".s64", "pk_total"), (".s64", "pk_csize")]
    for v in scalars:
      pty = _PTX_PARAM_TYPE.get(v.type.name)
      if pty is None:
        raise PtxUnsupported(f"unsupported scalar param type {v.type.name}")
      params.append((pty, f"p_s_{v.uid}"))
    params.append((".u64", "pk_flag"))
    params.append((".u64", "pk_info"))
    for v in bufs:
      if v.type.elt.name not in _SCALAR_REG_CLASS:
        raise PtxUnsupported(f"unsupported buffer element type {v.type.elt.name}")
      params.append((".u64", f"p_b_{v.uid}"))

    total_r = self.new_b64()
    csize_r = self.new_b64()
    self.emit(f"ld.param.s64 {total_r}, [pk_total];", defs=(total_r,))
    self.emit(f"ld.param.s64 {csize_r}, [pk_csize];", defs=(csize_r,))

    for v in scalars:
      pname = f"p_s_{v.uid}"
      reg = self.ensure_var_reg(v)
      if v.type.name == "bool":
        tmp = self.new_b16()
        self.emit(f"ld.param.u8 {tmp}, [{pname}];", defs=(tmp,))
        self.emit(f"setp.ne.u16 {reg}, {tmp}, 0;", defs=(reg,), uses=(tmp,))
      else:
        pty = _PTX_ARITH_TYPE[v.type.name]
        self.emit(f"ld.param.{pty} {reg}, [{pname}];", defs=(reg,))

    self.flag_reg = self.new_b64()
    self.info_reg = self.new_b64()
    self.emit(f"ld.param.u64 {self.flag_reg}, [pk_flag];", defs=(self.flag_reg,))
    self.emit(f"ld.param.u64 {self.info_reg}, [pk_info];", defs=(self.info_reg,))

    for v in bufs:
      reg = self.new_b64()
      self.buf_regs[v.uid] = reg
      self.emit(f"ld.param.u64 {reg}, [p_b_{v.uid}];", defs=(reg,))

    return params, total_r, csize_r

  def emit_shfl_down(self, src: str, delta: int, t: ScalarType) -> str:
    if t.itemsize == 4:
      if t.name == "float32":
        r_in, r_out = self.new_b32(), self.new_b32()
        dst = self.new_f32()
        self.emit(f"mov.b32 {r_in}, {src};", defs=(r_in,), uses=(src,))
        self.emit(
          f"shfl.sync.down.b32 {r_out}, {r_in}, {delta}, 31, 0xffffffff;",
          defs=(r_out,),
          uses=(r_in,),
          side_effect=True,
        )
        self.emit(f"mov.b32 {dst}, {r_out};", defs=(dst,), uses=(r_out,))
        return dst
      dst = self.new_reg(t)
      self.emit(
        f"shfl.sync.down.b32 {dst}, {src}, {delta}, 31, 0xffffffff;",
        defs=(dst,),
        uses=(src,),
        side_effect=True,
      )
      return dst
    if t.itemsize == 8:
      lo, hi = self.new_b32(), self.new_b32()
      lo2, hi2 = self.new_b32(), self.new_b32()
      dst = self.new_reg(t)
      self.emit(f"mov.b64 {{{lo}, {hi}}}, {src};", defs=(lo, hi), uses=(src,))
      self.emit(
        f"shfl.sync.down.b32 {lo2}, {lo}, {delta}, 31, 0xffffffff;",
        defs=(lo2,),
        uses=(lo,),
        side_effect=True,
      )
      self.emit(
        f"shfl.sync.down.b32 {hi2}, {hi}, {delta}, 31, 0xffffffff;",
        defs=(hi2,),
        uses=(hi,),
        side_effect=True,
      )
      self.emit(f"mov.b64 {dst}, {{{lo2}, {hi2}}};", defs=(dst,), uses=(lo2, hi2))
      return dst
    raise PtxUnsupported(f"shfl_down unsupported for {t.name}")

  def emit_shfl_bcast(self, src: str, src_lane: int, t: ScalarType) -> str:
    if t.itemsize == 4:
      if t.name == "float32":
        r_in, r_out = self.new_b32(), self.new_b32()
        dst = self.new_f32()
        self.emit(f"mov.b32 {r_in}, {src};", defs=(r_in,), uses=(src,))
        self.emit(
          f"shfl.sync.idx.b32 {r_out}, {r_in}, {src_lane}, 31, 0xffffffff;",
          defs=(r_out,),
          uses=(r_in,),
          side_effect=True,
        )
        self.emit(f"mov.b32 {dst}, {r_out};", defs=(dst,), uses=(r_out,))
        return dst
      dst = self.new_reg(t)
      self.emit(
        f"shfl.sync.idx.b32 {dst}, {src}, {src_lane}, 31, 0xffffffff;",
        defs=(dst,),
        uses=(src,),
        side_effect=True,
      )
      return dst
    if t.itemsize == 8:
      lo, hi = self.new_b32(), self.new_b32()
      lo2, hi2 = self.new_b32(), self.new_b32()
      dst = self.new_reg(t)
      self.emit(f"mov.b64 {{{lo}, {hi}}}, {src};", defs=(lo, hi), uses=(src,))
      self.emit(
        f"shfl.sync.idx.b32 {lo2}, {lo}, {src_lane}, 31, 0xffffffff;",
        defs=(lo2,),
        uses=(lo,),
        side_effect=True,
      )
      self.emit(
        f"shfl.sync.idx.b32 {hi2}, {hi}, {src_lane}, 31, 0xffffffff;",
        defs=(hi2,),
        uses=(hi,),
        side_effect=True,
      )
      self.emit(f"mov.b64 {dst}, {{{lo2}, {hi2}}};", defs=(dst,), uses=(lo2, hi2))
      return dst
    raise PtxUnsupported(f"shfl_bcast unsupported for {t.name}")

  def emit_combine(self, op: str, lhs: str, rhs: str, t: ScalarType) -> str:
    ty = _PTX_ARITH_TYPE[t.name]
    if op == "add":
      dst = self.new_reg(t)
      self.emit(f"add.{ty} {dst}, {lhs}, {rhs};", defs=(dst,), uses=(lhs, rhs))
      return dst
    if op == "mul":
      dst = self.new_reg(t)
      if t.kind == "f":
        self.emit(f"mul.{ty} {dst}, {lhs}, {rhs};", defs=(dst,), uses=(lhs, rhs))
      else:
        self.emit(f"mul.lo.{ty} {dst}, {lhs}, {rhs};", defs=(dst,), uses=(lhs, rhs))
      return dst
    if op in ("min", "max"):
      if t.kind == "f":
        p_an, p_bn, p_cmp = self.new_pred(), self.new_pred(), self.new_pred()
        t1, t2, dst = self.new_reg(t), self.new_reg(t), self.new_reg(t)
        cmp_op = "lt" if op == "min" else "gt"
        self.emit(f"setp.{cmp_op}.{ty} {p_cmp}, {lhs}, {rhs};", defs=(p_cmp,), uses=(lhs, rhs))
        self.emit(f"selp.{ty} {t1}, {lhs}, {rhs}, {p_cmp};", defs=(t1,), uses=(lhs, rhs, p_cmp))
        self.emit(f"setp.neu.{ty} {p_bn}, {rhs}, {rhs};", defs=(p_bn,), uses=(rhs,))
        self.emit(f"selp.{ty} {t2}, {rhs}, {t1}, {p_bn};", defs=(t2,), uses=(rhs, t1, p_bn))
        self.emit(f"setp.neu.{ty} {p_an}, {lhs}, {lhs};", defs=(p_an,), uses=(lhs,))
        self.emit(f"selp.{ty} {dst}, {lhs}, {t2}, {p_an};", defs=(dst,), uses=(lhs, t2, p_an))
        return dst
      dst = self.new_reg(t)
      self.emit(f"{op}.{ty} {dst}, {lhs}, {rhs};", defs=(dst,), uses=(lhs, rhs))
      return dst
    raise PtxUnsupported(f"unsupported warp-reduce combine op {op}")


# ---------------------------------------------------------------------------
# Entry generators (plain ParLoop and WarpRowReducePlan)
# ---------------------------------------------------------------------------


def _emit_plain_parloop_entry(plan: KernelPlan, entry_name: str) -> PtxEntry:
  loop = plan.loop
  if loop.shared_tile is not None or loop.monoidal_tile is not None or loop.warp_reduce is not None:
    raise PtxUnsupported("specialized tiled/monoidal/warp loop")

  em = PtxEmitter()
  params, total_r, csize_r = em.load_params(plan.scalars, plan.bufs)

  # chunk = (long long)blockIdx.x * blockDim.x + threadIdx.x
  r_cta, r_ntid, r_tid = em.new_b32(), em.new_b32(), em.new_b32()
  rd_prod, rd_tid = em.new_b64(), em.new_b64()
  chunk_r = em.ensure_var_reg(loop.chunk)
  start_r = em.ensure_var_reg(loop.start)
  stop_r = em.ensure_var_reg(loop.stop)

  em.emit(f"mov.u32 {r_cta}, %ctaid.x;", defs=(r_cta,))
  em.emit(f"mov.u32 {r_ntid}, %ntid.x;", defs=(r_ntid,))
  em.emit(f"mov.u32 {r_tid}, %tid.x;", defs=(r_tid,))
  em.emit(f"mul.wide.u32 {rd_prod}, {r_cta}, {r_ntid};", defs=(rd_prod,), uses=(r_cta, r_ntid))
  em.emit(f"cvt.u64.u32 {rd_tid}, {r_tid};", defs=(rd_tid,), uses=(r_tid,))
  em.emit(f"add.s64 {chunk_r}, {rd_prod}, {rd_tid};", defs=(chunk_r,), uses=(rd_prod, rd_tid))
  em.emit(f"mul.lo.s64 {start_r}, {chunk_r}, {csize_r};", defs=(start_r,), uses=(chunk_r, csize_r))

  p_oob = em.new_pred()
  l_run = em.new_label("L_run")
  em.emit(f"setp.lt.s64 {p_oob}, {start_r}, {total_r};", defs=(p_oob,), uses=(start_r, total_r))
  em.emit(f"@{p_oob} bra {l_run};", uses=(p_oob,), side_effect=True)
  em.emit("ret;", side_effect=True)
  em.emit_label(l_run)

  rd_end = em.new_b64()
  em.emit(f"add.s64 {rd_end}, {start_r}, {csize_r};", defs=(rd_end,), uses=(start_r, csize_r))
  em.emit(f"min.s64 {stop_r}, {rd_end}, {total_r};", defs=(stop_r,), uses=(rd_end, total_r))

  # Zero-initialize every defined local scalar (matches CudaPlanner semantics)
  defined = defined_vars(loop.body, into_parloops=True)
  for _uid, v in sorted(defined.items()):
    if v is loop.chunk or v is loop.start or v is loop.stop:
      continue
    if isinstance(v.type, BufType):
      raise PtxUnsupported("local buffer inside plain kernel")
    r = em.ensure_var_reg(v)
    em.emit_zero_reg(r, v.type)

  em.block(loop.body)
  em.emit("ret;", side_effect=True)

  if em.errors != plan.errors:
    raise PtxUnsupported("error ID mismatch with CUDA plan")
  return PtxEntry(name=entry_name, params=params, reg_types=em.reg_types, insts=em.insts)


def _emit_warp_reduce_entry(plan: KernelPlan, entry_name: str) -> PtxEntry:
  loop = plan.loop
  wplan = loop.warp_reduce
  if wplan is None or not wplan.stages:
    raise PtxUnsupported("missing warp_reduce plan")

  em = PtxEmitter()
  params, _total_r, _csize_r = em.load_params(plan.scalars, plan.bufs)

  r_cta, r_tid = em.new_b32(), em.new_b32()
  blk_r = em.new_b64()
  wid_r = em.new_b64()
  lane32_r = em.new_b32()
  lane64_r = em.new_b64()
  r_wid32 = em.new_b32()

  em.emit(f"mov.u32 {r_cta}, %ctaid.x;", defs=(r_cta,))
  em.emit(f"mov.u32 {r_tid}, %tid.x;", defs=(r_tid,))
  em.emit(f"cvt.u64.u32 {blk_r}, {r_cta};", defs=(blk_r,), uses=(r_cta,))
  em.emit(f"shr.u32 {r_wid32}, {r_tid}, 5;", defs=(r_wid32,), uses=(r_tid,))
  em.emit(f"cvt.u64.u32 {wid_r}, {r_wid32};", defs=(wid_r,), uses=(r_wid32,))
  em.emit(f"and.b32 {lane32_r}, {r_tid}, 31;", defs=(lane32_r,), uses=(r_tid,))
  em.emit(f"cvt.u64.u32 {lane64_r}, {lane32_r};", defs=(lane64_r,), uses=(lane32_r,))

  n_div_8 = False
  k_nonempty = False
  k_unroll_iters = 0
  if wplan.dim_hints is not None:
    nh, kh = wplan.dim_hints
    n_div_8 = nh.divisible_by(8)
    k_nonempty = kh.lo >= 1
    if kh.exact is not None and 32 <= kh.exact <= 256 and kh.exact % 32 == 0:
      k_unroll_iters = kh.exact // 32

  n_r = em._to_i64(em.expr(wplan.shape[-1]), wplan.shape[-1].type)
  k_r = em._to_i64(em.expr(wplan.k_dim), wplan.k_dim.type)

  # pk_nb_n = (pk_N + 7) / 8; pk_blk_n = pk_blk % pk_nb_n; idx[-1] = pk_blk_n * 8 + pk_wid
  n_p7, nb_n = em.new_b64(), em.new_b64()
  blk_n, blk_n8 = em.new_b64(), em.new_b64()
  em.emit(f"add.s64 {n_p7}, {n_r}, 7;", defs=(n_p7,), uses=(n_r,))
  em.emit(f"shr.s64 {nb_n}, {n_p7}, 3;", defs=(nb_n,), uses=(n_p7,))
  em.emit(f"rem.s64 {blk_n}, {blk_r}, {nb_n};", defs=(blk_n,), uses=(blk_r, nb_n))
  em.emit(f"shl.b64 {blk_n8}, {blk_n}, 3;", defs=(blk_n8,), uses=(blk_n,))
  last_idx_r = em.ensure_var_reg(wplan.index[-1])
  em.emit(f"add.s64 {last_idx_r}, {blk_n8}, {wid_r};", defs=(last_idx_r,), uses=(blk_n8, wid_r))

  if not n_div_8:
    p_in = em.new_pred()
    l_in = em.new_label("L_row_in")
    em.emit(f"setp.lt.s64 {p_in}, {last_idx_r}, {n_r};", defs=(p_in,), uses=(last_idx_r, n_r))
    em.emit(f"@{p_in} bra {l_in};", uses=(p_in,), side_effect=True)
    em.emit("ret;", side_effect=True)
    em.emit_label(l_in)

  if len(wplan.shape) >= 2:
    rem_r = em.new_b64()
    em.emit(f"div.s64 {rem_r}, {blk_r}, {nb_n};", defs=(rem_r,), uses=(blk_r, nb_n))
    for ax in range(len(wplan.shape) - 2, 0, -1):
      d_r = em._to_i64(em.expr(wplan.shape[ax]), wplan.shape[ax].type)
      iv_r = em.ensure_var_reg(wplan.index[ax])
      next_rem = em.new_b64()
      em.emit(f"rem.s64 {iv_r}, {rem_r}, {d_r};", defs=(iv_r,), uses=(rem_r, d_r))
      em.emit(f"div.s64 {next_rem}, {rem_r}, {d_r};", defs=(next_rem,), uses=(rem_r, d_r))
      rem_r = next_rem
    idx0_r = em.ensure_var_reg(wplan.index[0])
    em.emit(f"mov.b64 {idx0_r}, {rem_r};", defs=(idx0_r,), uses=(rem_r,))

  empty_errs = [f.empty for st in wplan.stages for f in st.reduce.folds if f.empty is not None]
  if empty_errs and not k_nonempty:
    eid = em.error_id(empty_errs[0])
    p_kpos = em.new_pred()
    p_l0 = em.new_pred()
    l_nonempty = em.new_label("L_k_pos")
    l_empty_ret = em.new_label("L_k_empty_ret")
    em.emit(f"setp.gt.s64 {p_kpos}, {k_r}, 0;", defs=(p_kpos,), uses=(k_r,))
    em.emit(f"@{p_kpos} bra {l_nonempty};", uses=(p_kpos,), side_effect=True)
    em.emit(f"setp.ne.u32 {p_l0}, {lane32_r}, 0;", defs=(p_l0,), uses=(lane32_r,))
    em.emit(f"@{p_l0} bra {l_empty_ret};", uses=(p_l0,), side_effect=True)
    em._emit_fail_body(eid, [], l_empty_ret)
    em.emit_label(l_nonempty)

  declared_uids: set[int] = {v.uid for v in wplan.index}

  def _init_locals(stmts: list[ir.Stmt], extra_skip: set[int] = frozenset()) -> None:
    for uid, v in sorted(defined_vars(stmts, into_parloops=True).items()):
      if uid in declared_uids or uid in extra_skip:
        continue
      if isinstance(v.type, BufType):
        raise PtxUnsupported("local buffer in warp_reduce")
      r = em.ensure_var_reg(v)
      em.emit_zero_reg(r, v.type)

  for stage in wplan.stages:
    red = stage.reduce
    folds = red.folds
    if stage.before:
      _init_locals(stage.before)
      em.block(stage.before)
      declared_uids |= set(defined_vars(stage.before))

    has_unseeded = any(f.op in ("max", "min") for f in folds)
    acc_regs: list[str] = []
    for f in folds:
      ar = em.new_reg(f.a.type)
      if f.op == "mul":
        em.emit_const_into(ar, 1, f.a.type)
      else:
        em.emit_zero_reg(ar, f.a.type)
      acc_regs.append(ar)

    has_r = em.new_b32() if has_unseeded else ""
    if has_unseeded:
      em.emit(f"mov.b32 {has_r}, 0;", defs=(has_r,))

    k_var = red.index[0]
    k_var_r = em.ensure_var_reg(k_var)
    _init_locals(red.body, {k_var.uid})

    def _emit_fold_iter():
      em.block(red.body)
      for fi, f in enumerate(folds):
        if f.value.type.name != f.a.type.name:
          ve = em._emit_cast(ir.Cast(f.value, f.a.type))
        else:
          ve = em.expr(f.value)
        if f.op in ("add", "mul"):
          comb = em.emit_combine(f.op, acc_regs[fi], ve, f.a.type)
          em.emit_mov(acc_regs[fi], comb, f.a.type)
        else:
          comb = em.emit_combine(f.op, acc_regs[fi], ve, f.a.type)
          p_has = em.new_pred()
          sel = em.new_reg(f.a.type)
          ty = _PTX_ARITH_TYPE[f.a.type.name]
          em.emit(f"setp.ne.u32 {p_has}, {has_r}, 0;", defs=(p_has,), uses=(has_r,))
          em.emit(f"selp.{ty} {sel}, {comb}, {ve}, {p_has};", defs=(sel,), uses=(comb, ve, p_has))
          em.emit_mov(acc_regs[fi], sel, f.a.type)
      if has_unseeded:
        em.emit(f"mov.b32 {has_r}, 1;", defs=(has_r,))

    if k_unroll_iters > 0:
      for ki in range(k_unroll_iters):
        if ki == 0:
          em.emit(f"mov.b64 {k_var_r}, {lane64_r};", defs=(k_var_r,), uses=(lane64_r,))
        else:
          em.emit(f"add.s64 {k_var_r}, {lane64_r}, {ki * 32};", defs=(k_var_r,), uses=(lane64_r,))
        _emit_fold_iter()
    else:
      em.emit(f"mov.b64 {k_var_r}, {lane64_r};", defs=(k_var_r,), uses=(lane64_r,))
      l_kh = em.new_label("L_wk_head")
      l_ke = em.new_label("L_wk_end")
      em.emit_label(l_kh)
      p_kd = em.new_pred()
      em.emit(f"setp.ge.s64 {p_kd}, {k_var_r}, {k_r};", defs=(p_kd,), uses=(k_var_r, k_r))
      em.emit(f"@{p_kd} bra {l_ke};", uses=(p_kd,), side_effect=True)
      _emit_fold_iter()
      em.emit(f"add.s64 {k_var_r}, {k_var_r}, 32;", defs=(k_var_r,), uses=(k_var_r,))
      em.emit(f"bra {l_kh};", side_effect=True)
      em.emit_label(l_ke)

    # 5-step unrolled butterfly warp shuffle reduction (16, 8, 4, 2, 1)
    int32_t = ScalarType("int32")
    for off in (16, 8, 4, 2, 1):
      ohas_r = em.emit_shfl_down(has_r, off, int32_t) if has_unseeded else ""
      for fi, f in enumerate(folds):
        other = em.emit_shfl_down(acc_regs[fi], off, f.a.type)
        if f.op in ("add", "mul"):
          comb = em.emit_combine(f.op, acc_regs[fi], other, f.a.type)
          em.emit_mov(acc_regs[fi], comb, f.a.type)
        else:
          comb = em.emit_combine(f.op, acc_regs[fi], other, f.a.type)
          p_has, p_ohas = em.new_pred(), em.new_pred()
          cand, sel = em.new_reg(f.a.type), em.new_reg(f.a.type)
          ty = _PTX_ARITH_TYPE[f.a.type.name]
          em.emit(f"setp.ne.u32 {p_has}, {has_r}, 0;", defs=(p_has,), uses=(has_r,))
          em.emit(f"selp.{ty} {cand}, {comb}, {other}, {p_has};", defs=(cand,), uses=(comb, other, p_has))
          em.emit(f"setp.ne.u32 {p_ohas}, {ohas_r}, 0;", defs=(p_ohas,), uses=(ohas_r,))
          em.emit(f"selp.{ty} {sel}, {cand}, {acc_regs[fi]}, {p_ohas};", defs=(sel,), uses=(cand, acc_regs[fi], p_ohas))
          em.emit_mov(acc_regs[fi], sel, f.a.type)
      if has_unseeded:
        em.emit(f"or.b32 {has_r}, {has_r}, {ohas_r};", defs=(has_r,), uses=(has_r, ohas_r))

    # Declare broadcast vars at warp scope
    for bv in stage.broadcast_vars:
      if bv.uid not in declared_uids:
        r = em.ensure_var_reg(bv)
        em.emit_zero_reg(r, bv.type)
        declared_uids.add(bv.uid)

    # Lane 0 applies non-identity init and runs stage.after
    p_not_l0 = em.new_pred()
    l_after_l0 = em.new_label("L_after_l0")
    em.emit(f"setp.ne.u32 {p_not_l0}, {lane32_r}, 0;", defs=(p_not_l0,), uses=(lane32_r,))
    em.emit(f"@{p_not_l0} bra {l_after_l0};", uses=(p_not_l0,), side_effect=True)

    fold_res_uids: set[int] = set()
    for fi, f in enumerate(folds):
      ar = acc_regs[fi]
      if f.init is not None:
        is_triv_id = isinstance(f.init, ir.Const) and (
          (f.op == "add" and f.init.value == 0) or (f.op == "mul" and f.init.value == 1)
        )
        if not is_triv_id:
          init_r = em.expr(f.init)
          if f.init.type.name != f.a.type.name:
            init_r = em._emit_cast(ir.Cast(f.init, f.a.type))
          if f.op in ("add", "mul"):
            comb = em.emit_combine(f.op, init_r, ar, f.a.type)
            em.emit_mov(ar, comb, f.a.type)
          else:
            comb = em.emit_combine(f.op, init_r, ar, f.a.type)
            p_has = em.new_pred()
            sel = em.new_reg(f.a.type)
            ty = _PTX_ARITH_TYPE[f.a.type.name]
            em.emit(f"setp.ne.u32 {p_has}, {has_r}, 0;", defs=(p_has,), uses=(has_r,))
            em.emit(f"selp.{ty} {sel}, {comb}, {init_r}, {p_has};", defs=(sel,), uses=(comb, init_r, p_has))
            em.emit_mov(ar, sel, f.a.type)
      res_r = em.ensure_var_reg(f.result)
      em.emit_mov(res_r, ar, f.a.type)
      fold_res_uids.add(f.result.uid)

    _init_locals(stage.after, fold_res_uids)
    em.block(stage.after)
    em.emit_label(l_after_l0)

    for bv in stage.broadcast_vars:
      br = em.ensure_var_reg(bv)
      bcast = em.emit_shfl_bcast(br, 0, bv.type)
      em.emit_mov(br, bcast, bv.type)

  if wplan.epilogue is not None:
    ep = wplan.epilogue
    ep_k = ep.index[0]
    ep_k_r = em.ensure_var_reg(ep_k)
    ep_body = [t for t in ep.body if not (isinstance(t, ir.ReleaseScope) and t.scope is ep.scope)]
    _init_locals(ep_body, {ep_k.uid})
    if k_unroll_iters > 0:
      for ki in range(k_unroll_iters):
        if ki == 0:
          em.emit(f"mov.b64 {ep_k_r}, {lane64_r};", defs=(ep_k_r,), uses=(lane64_r,))
        else:
          em.emit(f"add.s64 {ep_k_r}, {lane64_r}, {ki * 32};", defs=(ep_k_r,), uses=(lane64_r,))
        em.block(ep_body)
    else:
      em.emit(f"mov.b64 {ep_k_r}, {lane64_r};", defs=(ep_k_r,), uses=(lane64_r,))
      l_eh = em.new_label("L_ep_head")
      l_ee = em.new_label("L_ep_end")
      em.emit_label(l_eh)
      p_ed = em.new_pred()
      em.emit(f"setp.ge.s64 {p_ed}, {ep_k_r}, {k_r};", defs=(p_ed,), uses=(ep_k_r, k_r))
      em.emit(f"@{p_ed} bra {l_ee};", uses=(p_ed,), side_effect=True)
      em.block(ep_body)
      em.emit(f"add.s64 {ep_k_r}, {ep_k_r}, 32;", defs=(ep_k_r,), uses=(ep_k_r,))
      em.emit(f"bra {l_eh};", side_effect=True)
      em.emit_label(l_ee)

  em.emit("ret;", side_effect=True)
  if em.errors != plan.errors:
    raise PtxUnsupported("error ID mismatch with CUDA plan")
  return PtxEntry(name=entry_name, params=params, reg_types=em.reg_types, insts=em.insts)


# ---------------------------------------------------------------------------
# PtxTidy: use-count dead-register pruning (fixed-point)
# ---------------------------------------------------------------------------


def ptx_tidy(entry: PtxEntry) -> PtxEntry:
  """Prune dead pure instructions and unused virtual registers to fixed point."""
  insts = list(entry.insts)
  while True:
    use_counts: dict[str, int] = {}
    for ins in insts:
      for r in ins.uses:
        use_counts[r] = use_counts.get(r, 0) + 1
    kept: list[PtxInst] = []
    removed = 0
    for ins in insts:
      if not ins.side_effect and ins.defs and all(use_counts.get(d, 0) == 0 for d in ins.defs):
        removed += 1
        continue
      kept.append(ins)
    insts = kept
    if removed == 0:
      break

  live_regs: set[str] = set()
  for ins in insts:
    live_regs.update(ins.defs)
    live_regs.update(ins.uses)
  reg_types = {r: t for r, t in entry.reg_types.items() if r in live_regs}
  return PtxEntry(name=entry.name, params=entry.params, reg_types=reg_types, insts=insts)


def format_ptx_entry(entry: PtxEntry) -> str:
  """Serialize one :class:`PtxEntry` to a ``.visible .entry`` PTX block."""
  param_lines = [f"  .param {pty} {pname}" for pty, pname in entry.params]
  lines = [f".visible .entry {entry.name}(", ",\n".join(param_lines), ")", "{"]

  # Group declared registers by PTX type for compact declarations
  by_type: dict[str, list[str]] = {}
  for rname, rty in entry.reg_types.items():
    by_type.setdefault(rty, []).append(rname)
  for rty in (".pred", ".b16", ".b32", ".b64", ".f32", ".f64"):
    regs = by_type.get(rty)
    if regs:
      lines.append(f"  .reg {rty} {', '.join(regs)};")

  for ins in entry.insts:
    if ins.is_label:
      lines.append(ins.text)
    else:
      lines.append(f"  {ins.text}")
  lines.append("}")
  return "\n".join(lines)


def bundle_ptx_module(entry_bodies: list[str], arch: str = "sm_80", version: str = "7.8") -> str:
  """Wrap one or more serialized ``.visible .entry`` bodies into a PTX module."""
  header = f".version {version}\n.target {arch}\n.address_size 64\n\n"
  return header + "\n\n".join(entry_bodies) + "\n"


def emit_ptx_entry_body(plan: KernelPlan, entry_name: str = "pk_kernel") -> str | None:
  """Try to emit a tidied ``.visible .entry`` body for ``plan``; return ``None`` on fallback."""
  try:
    if plan.loop.warp_reduce is not None:
      entry = _emit_warp_reduce_entry(plan, entry_name)
    elif plan.loop.shared_tile is None and plan.loop.monoidal_tile is None:
      entry = _emit_plain_parloop_entry(plan, entry_name)
    else:
      return None
  except PtxUnsupported:
    return None
  return format_ptx_entry(ptx_tidy(entry))


def emit_ptx_kernel(plan: KernelPlan, entry_name: str = "pk_kernel", arch: str = "sm_80") -> str | None:
  """Emit a complete standalone PTX 7.8 module for ``plan``, or ``None`` if unsupported."""
  body = emit_ptx_entry_body(plan, entry_name=entry_name)
  if body is None:
    return None
  return bundle_ptx_module([body], arch=arch)


# ---------------------------------------------------------------------------
# PtxCheck: offline static PTX validator (no GPU required)
# ---------------------------------------------------------------------------

_SPECIAL_REGS = frozenset(
  {
    "%ctaid.x",
    "%ctaid.y",
    "%ctaid.z",
    "%ntid.x",
    "%ntid.y",
    "%ntid.z",
    "%tid.x",
    "%tid.y",
    "%tid.z",
    "%laneid",
    "%warpid",
  }
)
_REG_RE = re.compile(r"%[a-zA-Z_][a-zA-Z0-9_.]*")
_PARAM_DECL_RE = re.compile(r"\.param\s+(\.[a-z0-9]+)\s+([a-zA-Z_][a-zA-Z0-9_]*)")
_REG_DECL_RE = re.compile(r"^\.reg\s+(\.[a-z0-9]+)\s+([^;]+);$")
_LD_PARAM_RE = re.compile(r"ld\.param\.[a-z0-9]+\s+%[a-zA-Z0-9_]+,\s*\[([a-zA-Z_][a-zA-Z0-9_]*)\]")
_BRA_RE = re.compile(r"\bbra\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;")
_LABEL_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*):$")
_ENTRY_RE = re.compile(r"^\.visible\s+\.entry\s+([a-zA-Z_][a-zA-Z0-9_]*)\(")


def ptx_check(ptx_source: str) -> list[str]:
  """Statically validate generated PTX source; returns a list of error strings (empty = valid)."""
  errors: list[str] = []
  lines = [ln.strip() for ln in ptx_source.splitlines()]

  has_ver = any(ln.startswith(".version ") for ln in lines)
  has_tgt = any(ln.startswith(".target ") for ln in lines)
  has_addr = any(ln == ".address_size 64" for ln in lines)
  if not has_ver:
    errors.append("missing .version directive")
  if not has_tgt:
    errors.append("missing .target directive")
  if not has_addr:
    errors.append("missing .address_size 64 directive")

  i = 0
  n = len(lines)
  n_entries = 0
  while i < n:
    ln = lines[i]
    m = _ENTRY_RE.match(ln)
    if not m:
      i += 1
      continue
    n_entries += 1
    entry_name = m.group(1)
    # Collect parameter list until ')'
    declared_params: set[str] = set()
    while i < n and ")" not in lines[i]:
      pm = _PARAM_DECL_RE.search(lines[i])
      if pm:
        declared_params.add(pm.group(2))
      i += 1
    if i < n:
      pm = _PARAM_DECL_RE.search(lines[i])
      if pm:
        declared_params.add(pm.group(2))
      i += 1
    while i < n and lines[i] == "":
      i += 1
    if i >= n or lines[i] != "{":
      errors.append(f"{entry_name}: missing opening '{{'")
      break
    i += 1

    declared_regs: dict[str, str] = {}
    defined_labels: set[str] = set()
    used_labels: list[str] = []
    entry_insts: list[str] = []

    depth = 1
    while i < n and depth > 0:
      body_ln = lines[i]
      if body_ln == "{":
        depth += 1
      elif body_ln == "}":
        depth -= 1
        if depth == 0:
          i += 1
          break
      elif body_ln and not body_ln.startswith("//"):
        rm = _REG_DECL_RE.match(body_ln)
        if rm:
          rty = rm.group(1)
          for rtok in rm.group(2).split(","):
            rname = rtok.strip()
            if rname:
              declared_regs[rname] = rty
        else:
          lm = _LABEL_RE.match(body_ln)
          if lm:
            defined_labels.add(lm.group(1))
          else:
            entry_insts.append(body_ln)
      i += 1
    if depth != 0:
      errors.append(f"{entry_name}: unbalanced braces")

    for inst in entry_insts:
      if not inst.endswith(";"):
        errors.append(f"{entry_name}: instruction missing ';': {inst}")
      for reg in _REG_RE.findall(inst):
        if reg in _SPECIAL_REGS:
          continue
        if reg not in declared_regs:
          errors.append(f"{entry_name}: undeclared register {reg} in: {inst}")
      lpm = _LD_PARAM_RE.search(inst)
      if lpm and lpm.group(1) not in declared_params:
        errors.append(f"{entry_name}: undeclared param {lpm.group(1)} in: {inst}")
      bm = _BRA_RE.search(inst)
      if bm:
        used_labels.append(bm.group(1))

      # Opcode sanity rules
      op_tok = inst.split()[1] if inst.startswith("@") else inst.split()[0]
      if op_tok in ("mul.s32", "mul.u32", "mul.s64", "mul.u64"):
        errors.append(f"{entry_name}: integer mul requires .lo/.hi/.wide: {inst}")
      if op_tok in ("div.f32", "div.f64", "sqrt.f32", "sqrt.f64"):
        errors.append(f"{entry_name}: float {op_tok} requires rounding mode (.rn/.approx): {inst}")
      if op_tok.startswith(("and.", "or.", "xor.", "not.")):
        suffix = op_tok.split(".")[-1]
        if suffix not in ("pred", "b16", "b32", "b64"):
          errors.append(f"{entry_name}: bitwise op {op_tok} requires .pred/.b* type: {inst}")

    for lbl in used_labels:
      if lbl not in defined_labels:
        errors.append(f"{entry_name}: branch to undefined label {lbl}")

  if n_entries == 0:
    errors.append("no .visible .entry found")
  return errors


def verify_ptx(ptx_source: str) -> None:
  """Raise :class:`PtxCheckError` if ``ptx_source`` fails static validation."""
  errs = ptx_check(ptx_source)
  if errs:
    raise PtxCheckError("\n".join(errs[:10]))
