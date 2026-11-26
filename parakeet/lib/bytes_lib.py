"""Compiled byte / ASCII parsing helpers over ``uint8`` arrays and ``RaggedArray[uint8]``.

Provides zero-``PyObject`` delimiter splitting (``split_bytes`` -> ``RaggedArray[uint8]``)
and compiled ASCII integer / floating-point parsers (``parse_int64``, ``parse_float32``,
``parse_float64``) that work both inside ``@parakeet.jit`` (parallel multicore / GPU)
and directly from ordinary Python.
"""

from __future__ import annotations

import numpy as np

from ..errors import TypeInferenceError
from ..ndtypes import Float32, Float64, Int64, UInt8
from ..ragged import RaggedArray
from ..syntax import ops
from ..syntax.ir import Assign, Break, Const, Expr, ForRange, If, ReleaseScope, Scope, Store, Var, While
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, imul, isub
from ..syntax.values import ArrayVal, describe
from .adverbs import LibFunction
from .core import register


def _as_sep_byte(el, sep) -> Expr:
  if isinstance(sep, str):
    if len(sep) != 1:
      raise el.error(TypeInferenceError, f"sep must be a single byte or character, got {sep!r}")
    return Const(ord(sep), UInt8)
  if isinstance(sep, ArrayVal) and sep.elt is UInt8 and sep.rank == 1 and isinstance(sep.shape[0], Const):
    if sep.shape[0].value != 1:
      raise el.error(TypeInferenceError, f"sep must have length 1, got {sep.shape[0].value}")
    return el.tmp(el.elem(sep, (ZERO,)), "sep")
  return el.tmp(cast(el.as_scalar(sep, "sep"), UInt8), "sep")


def _split_bytes_impl(el, buf, sep=10, skip_empty=False):
  """Split a 1-D ``uint8`` buffer on byte delimiter ``sep`` into a ``RaggedArray[uint8]``."""
  from . import numpy_lib as npl

  arr = el.as_array(buf, "split_bytes() buffer")
  if arr.rank != 1 or arr.elt is not UInt8:
    raise el.error(TypeInferenceError, f"split_bytes() expects a 1-d uint8/bytes buffer, got {describe(arr)}")
  arr = el.as_concrete(arr)
  sep_b = _as_sep_byte(el, sep)
  drop_empty = bool(el.static_bool(skip_empty, "skip_empty")) if not isinstance(skip_empty, bool) else skip_empty

  n = arr.shape[0]
  # Pass 1: compact all non-separator bytes (data-parallel IndexFilter)
  is_sep = el.delayed(ops.Bool, (n,), lambda idx: cmp("eq", el.elem(arr, idx), sep_b), cost=1)
  is_keep = el.delayed(ops.Bool, (n,), lambda idx: cmp("ne", el.elem(arr, idx), sep_b), cost=1)
  values = el.as_concrete(el.getitem(arr, is_keep))

  # Pass 2: collect separator positions (IndexFilter over range(n))
  idx_arr = el.delayed(I, (n,), lambda idx: idx[0], cost=0)
  sep_pos = el.as_concrete(el.getitem(idx_arr, is_sep))
  m = sep_pos.shape[0]

  # Build per-segment lengths (m + 1 segments when n > 0; 0 segments when n == 0)
  n_segs_raw = el.tmp(ops.select(cmp("eq", n, ZERO), ZERO, iadd(m, ONE)), "nsegs")
  el.nonneg.add(n_segs_raw.uid)

  def _seg_len(idx):
    k = idx[0]
    start = ops.select(cmp("eq", k, ZERO), ZERO, iadd(el.elem(sep_pos, (isub(k, ONE),)), ONE))
    stop = ops.select(cmp("eq", k, m), n, el.elem(sep_pos, (k,)))
    return isub(stop, start)

  lengths = el.delayed(I, (n_segs_raw,), _seg_len, cost=3)
  if drop_empty:
    non_empty = el.delayed(ops.Bool, (n_segs_raw,), lambda idx: cmp("gt", el.elem(lengths, idx), ZERO), cost=2)
    lengths = el.as_concrete(el.getitem(lengths, non_empty))
  else:
    lengths = el.as_concrete(lengths)

  # Prefix sum lengths -> CSR offsets[0..num_segs]
  n_segs = lengths.shape[0]
  offsets = el.alloc_array(Int64, (el.tmp(iadd(n_segs, ONE), "noff"),), zero=False, label="sb_offs")
  el.emit(Store(offsets.buf, ZERO, ZERO))
  cs = npl.cumsum(el, lengths, axis=0)
  cs = el.as_concrete(cs)
  iv = Var("si", I)
  el.nonneg.add(iv.uid)
  el.upper[iv.uid] = n_segs
  sc = Scope("loop")
  blk: list = [
    Store(offsets.buf, iadd(iv, ONE), el.elem(cs, (iv,))),
    ReleaseScope(sc),
  ]
  el.emit(ForRange(iv, ZERO, n_segs, ONE, blk, sc, simd=True))
  el.record_store(offsets)
  return RaggedArray(values, offsets)


def _emit_while(el, cond: Expr, body: list, scope: Scope) -> None:
  el.emit(While([If(ops.lnot(cond), [Break()], []), *body], scope))


def _emit_parse_int64_slice(el, values: ArrayVal, start: Expr, stop: Expr) -> Expr:
  """Emit an unboxed ASCII signed int64 parser over ``values[start:stop]``."""
  pos = Var("p", I)
  acc = Var("acc", Int64)
  neg = Var("neg", ops.Bool)
  for v in (pos, acc, neg):
    el.mutables.add(v.uid)
  el.nonneg.add(pos.uid)
  el.emit(Assign(pos, start))
  el.emit(Assign(acc, Const(0, Int64)))
  el.emit(Assign(neg, ops.FALSE))

  # Skip leading ASCII whitespace (' ', '\t', '\r', '\n')
  ws_sc = Scope("loop")
  ws_blk: list = []
  with el.in_block(ws_blk, ws_sc, loop=True):
    ch = el.tmp(el.elem(values, (pos,)), "ch")
    is_ws = ops.lor(
      ops.lor(cmp("eq", ch, Const(32, UInt8)), cmp("eq", ch, Const(9, UInt8))),
      ops.lor(cmp("eq", ch, Const(10, UInt8)), cmp("eq", ch, Const(13, UInt8))),
    )
    el.emit(If(is_ws, [Assign(pos, iadd(pos, ONE))], [Break()]))
    el.emit(ReleaseScope(ws_sc))
  _emit_while(el, cmp("lt", pos, stop), ws_blk, ws_sc)

  # Optional sign '+' (43) or '-' (45)
  sign_blk: list = []
  with el.in_block(sign_blk):
    ch = el.tmp(el.elem(values, (pos,)), "sch")
    el.emit(
      If(
        cmp("eq", ch, Const(45, UInt8)),
        [Assign(neg, ops.TRUE), Assign(pos, iadd(pos, ONE))],
        [If(cmp("eq", ch, Const(43, UInt8)), [Assign(pos, iadd(pos, ONE))], [])],
      )
    )
  el.emit(If(cmp("lt", pos, stop), sign_blk, []))

  # Accumulate ASCII digits '0' (48) .. '9' (57)
  dig_sc = Scope("loop")
  dig_blk: list = []
  with el.in_block(dig_blk, dig_sc, loop=True):
    ch = el.tmp(el.elem(values, (pos,)), "dch")
    is_dig = ops.land(cmp("ge", ch, Const(48, UInt8)), cmp("le", ch, Const(57, UInt8)))
    digit = cast(ops.prim("sub", (ch, Const(48, UInt8)), UInt8), Int64)
    step = [
      Assign(acc, iadd(imul(acc, Const(10, Int64)), digit)),
      Assign(pos, iadd(pos, ONE)),
    ]
    el.emit(If(is_dig, step, [Break()]))
    el.emit(ReleaseScope(dig_sc))
  _emit_while(el, cmp("lt", pos, stop), dig_blk, dig_sc)

  return el.tmp(ops.select(neg, ops.prim("neg", (acc,), Int64), acc), "ival")


def _emit_parse_float_slice(el, values: ArrayVal, start: Expr, stop: Expr, out_elt) -> Expr:
  """Emit an unboxed ASCII float parser (``[+-]digits[.digits][eE[+-]digits]``)."""
  pos = Var("p", I)
  val = Var("fval", Float64)
  neg = Var("neg", ops.Bool)
  frac_scale = Var("fsc", Float64)
  exp_val = Var("expv", I)
  exp_neg = Var("eneg", ops.Bool)
  for v in (pos, val, neg, frac_scale, exp_val, exp_neg):
    el.mutables.add(v.uid)
  el.nonneg.add(pos.uid)
  el.emit(Assign(pos, start))
  el.emit(Assign(val, Const(0.0, Float64)))
  el.emit(Assign(neg, ops.FALSE))

  # Skip leading whitespace
  ws_sc = Scope("loop")
  ws_blk: list = []
  with el.in_block(ws_blk, ws_sc, loop=True):
    ch = el.tmp(el.elem(values, (pos,)), "ch")
    is_ws = ops.lor(
      ops.lor(cmp("eq", ch, Const(32, UInt8)), cmp("eq", ch, Const(9, UInt8))),
      ops.lor(cmp("eq", ch, Const(10, UInt8)), cmp("eq", ch, Const(13, UInt8))),
    )
    el.emit(If(is_ws, [Assign(pos, iadd(pos, ONE))], [Break()]))
    el.emit(ReleaseScope(ws_sc))
  _emit_while(el, cmp("lt", pos, stop), ws_blk, ws_sc)

  # Optional sign
  sign_blk: list = []
  with el.in_block(sign_blk):
    ch = el.tmp(el.elem(values, (pos,)), "sch")
    el.emit(
      If(
        cmp("eq", ch, Const(45, UInt8)),
        [Assign(neg, ops.TRUE), Assign(pos, iadd(pos, ONE))],
        [If(cmp("eq", ch, Const(43, UInt8)), [Assign(pos, iadd(pos, ONE))], [])],
      )
    )
  el.emit(If(cmp("lt", pos, stop), sign_blk, []))

  # Integer part
  int_sc = Scope("loop")
  int_blk: list = []
  with el.in_block(int_blk, int_sc, loop=True):
    ch = el.tmp(el.elem(values, (pos,)), "ich")
    is_dig = ops.land(cmp("ge", ch, Const(48, UInt8)), cmp("le", ch, Const(57, UInt8)))
    digit = cast(ops.prim("sub", (ch, Const(48, UInt8)), UInt8), Float64)
    step = [
      Assign(val, ops.prim("add", (ops.prim("mul", (val, Const(10.0, Float64)), Float64), digit), Float64)),
      Assign(pos, iadd(pos, ONE)),
    ]
    el.emit(If(is_dig, step, [Break()]))
    el.emit(ReleaseScope(int_sc))
  _emit_while(el, cmp("lt", pos, stop), int_blk, int_sc)

  # Optional fractional part '.' (46)
  dot_blk: list = []
  with el.in_block(dot_blk):
    ch = el.tmp(el.elem(values, (pos,)), "dot")
    frac_if_blk: list = []
    with el.in_block(frac_if_blk):
      el.emit(Assign(pos, iadd(pos, ONE)))
      el.emit(Assign(frac_scale, Const(0.1, Float64)))
      fr_sc = Scope("loop")
      fr_blk: list = []
      with el.in_block(fr_blk, fr_sc, loop=True):
        fch = el.tmp(el.elem(values, (pos,)), "fch")
        is_dig = ops.land(cmp("ge", fch, Const(48, UInt8)), cmp("le", fch, Const(57, UInt8)))
        digit = cast(ops.prim("sub", (fch, Const(48, UInt8)), UInt8), Float64)
        step = [
          Assign(val, ops.prim("add", (val, ops.prim("mul", (digit, frac_scale), Float64)), Float64)),
          Assign(frac_scale, ops.prim("mul", (frac_scale, Const(0.1, Float64)), Float64)),
          Assign(pos, iadd(pos, ONE)),
        ]
        el.emit(If(is_dig, step, [Break()]))
        el.emit(ReleaseScope(fr_sc))
      _emit_while(el, cmp("lt", pos, stop), fr_blk, fr_sc)
    el.emit(If(cmp("eq", ch, Const(46, UInt8)), frac_if_blk, []))
  el.emit(If(cmp("lt", pos, stop), dot_blk, []))

  # Optional exponent 'e' (101) or 'E' (69)
  exp_outer: list = []
  with el.in_block(exp_outer):
    ch = el.tmp(el.elem(values, (pos,)), "ech")
    is_e = ops.lor(cmp("eq", ch, Const(101, UInt8)), cmp("eq", ch, Const(69, UInt8)))
    exp_body: list = []
    with el.in_block(exp_body):
      el.emit(Assign(pos, iadd(pos, ONE)))
      el.emit(Assign(exp_neg, ops.FALSE))
      el.emit(Assign(exp_val, ZERO))
      esign_blk: list = []
      with el.in_block(esign_blk):
        es = el.tmp(el.elem(values, (pos,)), "esch")
        el.emit(
          If(
            cmp("eq", es, Const(45, UInt8)),
            [Assign(exp_neg, ops.TRUE), Assign(pos, iadd(pos, ONE))],
            [If(cmp("eq", es, Const(43, UInt8)), [Assign(pos, iadd(pos, ONE))], [])],
          )
        )
      el.emit(If(cmp("lt", pos, stop), esign_blk, []))
      ed_sc = Scope("loop")
      ed_blk: list = []
      with el.in_block(ed_blk, ed_sc, loop=True):
        ed = el.tmp(el.elem(values, (pos,)), "ed")
        is_dig = ops.land(cmp("ge", ed, Const(48, UInt8)), cmp("le", ed, Const(57, UInt8)))
        digit = cast(ops.prim("sub", (ed, Const(48, UInt8)), UInt8), I)
        el.emit(
          If(
            is_dig,
            [Assign(exp_val, iadd(imul(exp_val, Const(10, I)), digit)), Assign(pos, iadd(pos, ONE))],
            [Break()],
          )
        )
        el.emit(ReleaseScope(ed_sc))
      _emit_while(el, cmp("lt", pos, stop), ed_blk, ed_sc)
      signed_exp = ops.select(exp_neg, ops.prim("neg", (exp_val,), I), exp_val)
      scale = ops.prim("pow", (Const(10.0, Float64), cast(signed_exp, Float64)), Float64)
      el.emit(Assign(val, ops.prim("mul", (val, scale), Float64)))
    el.emit(If(is_e, exp_body, []))
  el.emit(If(cmp("lt", pos, stop), exp_outer, []))

  signed = ops.select(neg, ops.prim("neg", (val,), Float64), val)
  return el.tmp(cast(signed, out_elt), "fres")


def _parse_int64_impl(el, x):
  if isinstance(x, RaggedArray):
    vals = el.as_concrete(el.as_array(x.values, "parse_int64() values"))
    offs = el.as_concrete(el.as_array(x.offsets, "parse_int64() offsets"))
    if vals.rank != 1 or vals.elt is not UInt8:
      raise el.error(TypeInferenceError, f"parse_int64() expects RaggedArray[uint8], got {describe(vals)}")
    n_segs = el.tmp(isub(offs.shape[0], ONE), "nsegs")
    el.nonneg.add(n_segs.uid)
    out = el.alloc_array(Int64, (n_segs,), zero=False, label="p_i64")
    iv = Var("pi", I)
    el.nonneg.add(iv.uid)
    el.upper[iv.uid] = n_segs
    sc = Scope("loop")
    blk: list = []
    with el.in_block(blk, sc, loop=True, par=el.can_parallel((n_segs,))):
      s = el.tmp(cast(el.elem(offs, (iv,)), I), "ps")
      e = el.tmp(cast(el.elem(offs, (iadd(iv, ONE),)), I), "pe")
      v = _emit_parse_int64_slice(el, vals, s, e)
      el.emit(Store(out.buf, iv, v))
      el.emit(ReleaseScope(sc))
    el.emit(ForRange(iv, ZERO, n_segs, ONE, blk, sc, simd=True))
    el.record_store(out)
    return out
  arr = el.as_concrete(el.as_array(x, "parse_int64() buffer"))
  if arr.rank != 1 or arr.elt is not UInt8:
    raise el.error(TypeInferenceError, f"parse_int64() expects uint8 bytes or RaggedArray[uint8], got {describe(arr)}")
  return _emit_parse_int64_slice(el, arr, ZERO, arr.shape[0])


def _parse_float_impl(el, x, out_elt, fn_name: str):
  if isinstance(x, RaggedArray):
    vals = el.as_concrete(el.as_array(x.values, f"{fn_name}() values"))
    offs = el.as_concrete(el.as_array(x.offsets, f"{fn_name}() offsets"))
    if vals.rank != 1 or vals.elt is not UInt8:
      raise el.error(TypeInferenceError, f"{fn_name}() expects RaggedArray[uint8], got {describe(vals)}")
    n_segs = el.tmp(isub(offs.shape[0], ONE), "nsegs")
    el.nonneg.add(n_segs.uid)
    out = el.alloc_array(out_elt, (n_segs,), zero=False, label="p_flt")
    iv = Var("pi", I)
    el.nonneg.add(iv.uid)
    el.upper[iv.uid] = n_segs
    sc = Scope("loop")
    blk: list = []
    with el.in_block(blk, sc, loop=True, par=el.can_parallel((n_segs,))):
      s = el.tmp(cast(el.elem(offs, (iv,)), I), "ps")
      e = el.tmp(cast(el.elem(offs, (iadd(iv, ONE),)), I), "pe")
      v = _emit_parse_float_slice(el, vals, s, e, out_elt)
      el.emit(Store(out.buf, iv, v))
      el.emit(ReleaseScope(sc))
    el.emit(ForRange(iv, ZERO, n_segs, ONE, blk, sc, simd=True))
    el.record_store(out)
    return out
  arr = el.as_concrete(el.as_array(x, f"{fn_name}() buffer"))
  if arr.rank != 1 or arr.elt is not UInt8:
    raise el.error(TypeInferenceError, f"{fn_name}() expects uint8 bytes or RaggedArray[uint8], got {describe(arr)}")
  return _emit_parse_float_slice(el, arr, ZERO, arr.shape[0], out_elt)


# ---------------------------------------------------------------------------
# Python-callable wrappers (work both inside @par.jit and directly from Python)
# ---------------------------------------------------------------------------


def _py_split_bytes(buf, sep=10, skip_empty=False) -> RaggedArray:
  if isinstance(buf, (bytes, bytearray, memoryview)):
    arr = np.frombuffer(buf, dtype=np.uint8)
  else:
    arr = np.asarray(buf, dtype=np.uint8)
  if isinstance(sep, str):
    sep_val = ord(sep)
  elif isinstance(sep, (bytes, bytearray)) and len(sep) == 1:
    sep_val = sep[0]
  else:
    sep_val = int(sep)
  if arr.size == 0:
    return RaggedArray(np.empty(0, dtype=np.uint8), np.zeros(1, dtype=np.int64))
  sep_pos = np.nonzero(arr == sep_val)[0]
  values = arr[arr != sep_val]
  starts = np.empty(len(sep_pos) + 1, dtype=np.int64)
  starts[0] = 0
  starts[1:] = sep_pos + 1
  stops = np.empty(len(sep_pos) + 1, dtype=np.int64)
  stops[:-1] = sep_pos
  stops[-1] = len(arr)
  lengths = stops - starts
  if skip_empty:
    lengths = lengths[lengths > 0]
  offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
  np.cumsum(lengths, out=offsets[1:])
  return RaggedArray(values, offsets)


def _py_parse_int64(x):
  if isinstance(x, RaggedArray):
    vals = np.asarray(x.values, dtype=np.uint8)
    offs = np.asarray(x.offsets, dtype=np.int64)
    out = np.empty(len(offs) - 1, dtype=np.int64)
    for k in range(len(offs) - 1):
      tok = bytes(vals[offs[k] : offs[k + 1]]).strip()
      out[k] = np.int64(int(tok)) if tok else np.int64(0)
    return out
  b = bytes(x).strip()
  return np.int64(int(b)) if b else np.int64(0)


def _py_parse_float32(x):
  if isinstance(x, RaggedArray):
    vals = np.asarray(x.values, dtype=np.uint8)
    offs = np.asarray(x.offsets, dtype=np.int64)
    out = np.empty(len(offs) - 1, dtype=np.float32)
    for k in range(len(offs) - 1):
      tok = bytes(vals[offs[k] : offs[k + 1]]).strip()
      out[k] = np.float32(float(tok)) if tok else np.float32(0.0)
    return out
  b = bytes(x).strip()
  return np.float32(float(b)) if b else np.float32(0.0)


def _py_parse_float64(x):
  if isinstance(x, RaggedArray):
    vals = np.asarray(x.values, dtype=np.uint8)
    offs = np.asarray(x.offsets, dtype=np.int64)
    out = np.empty(len(offs) - 1, dtype=np.float64)
    for k in range(len(offs) - 1):
      tok = bytes(vals[offs[k] : offs[k + 1]]).strip()
      out[k] = np.float64(float(tok)) if tok else np.float64(0.0)
    return out
  b = bytes(x).strip()
  return np.float64(float(b)) if b else np.float64(0.0)


split_bytes = LibFunction("split_bytes", _split_bytes_impl, _py_split_bytes)
parse_int64 = LibFunction("parse_int64", _parse_int64_impl, _py_parse_int64)
parse_float32 = LibFunction(
  "parse_float32", lambda el, x: _parse_float_impl(el, x, Float32, "parse_float32"), _py_parse_float32
)
parse_float64 = LibFunction(
  "parse_float64", lambda el, x: _parse_float_impl(el, x, Float64, "parse_float64"), _py_parse_float64
)

register(split_bytes, name="split_bytes")(_split_bytes_impl)
register(parse_int64, name="parse_int64")(_parse_int64_impl)
register(parse_float32, name="parse_float32")(lambda el, x: _parse_float_impl(el, x, Float32, "parse_float32"))
register(parse_float64, name="parse_float64")(lambda el, x: _parse_float_impl(el, x, Float64, "parse_float64"))
