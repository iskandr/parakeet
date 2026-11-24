"""Segmented operators and packed variable-length FlashAttention over :class:`~parakeet.RaggedArray`."""

from __future__ import annotations

import math

from .. import prims
from ..errors import StaticValueError, TypeInferenceError
from ..ndtypes import Float32, Float64, Int64
from ..ragged import RaggedArray
from ..syntax import ops
from ..syntax.ir import Assign, Const, Expr, ForRange, If, Load, Store, Var
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, imax, imin, isub
from ..syntax.records import record_type
from ..syntax.values import ArrayVal, describe
from .adverbs import libfunction

_OP_ALIASES = {
  "add": "add",
  "sum": "add",
  "mul": "mul",
  "prod": "mul",
  "min": "min",
  "max": "max",
  "mean": "mean",
}


def _leading_dim(el, v) -> Expr:
  if isinstance(v, RaggedArray):
    return isub(v.offsets.shape[0], ONE)
  if isinstance(v, ArrayVal):
    return v.shape[0]
  if rec := record_type(v):
    vals = rec.values(v)
    return _leading_dim(el, vals[0]) if vals else ZERO
  raise el.error(TypeInferenceError, f"expected array or RaggedArray, got {describe(v)}")


def check_ragged_offsets(el, offsets: ArrayVal, total_len: Expr | None = None) -> ArrayVal:
  """Validate CSR offsets invariants once when ``bounds_check`` is enabled."""
  if not isinstance(offsets, ArrayVal) or offsets.rank != 1 or offsets.elt.kind not in "iu":
    raise el.error(TypeInferenceError, f"RaggedArray offsets must be a 1D integer array, got {describe(offsets)}")
  offsets = el.as_concrete(offsets)
  n_off = offsets.shape[0]
  if el.options.bounds_check:
    el.check(cmp("ge", n_off, ONE), "ValueError", "RaggedArray offsets must have length >= 1, got {0}", n_off)
    first = el.tmp(cast(el.elem(offsets, (ZERO,)), I), "off0")
    el.check(cmp("eq", first, ZERO), "ValueError", "RaggedArray offsets[0] must be 0, got {0}", first)
    n_seg = el.tmp(isub(n_off, ONE), "nseg")
    if total_len is not None:
      last = el.tmp(cast(el.elem(offsets, (n_seg,)), I), "off_last")
      el.check(
        cmp("le", last, total_len),
        "ValueError",
        "RaggedArray offsets[-1] ({0}) exceeds len(values) ({1})",
        last,
        total_len,
      )

    # Check monotonicity across all segments
    def _check_mono(idx):
      b = idx[0]
      s = cast(el.elem(offsets, (b,)), I)
      e = cast(el.elem(offsets, (iadd(b, ONE),)), I)
      el.check(cmp("le", s, e), "ValueError", "RaggedArray offsets must be non-decreasing ({0} > {1})", s, e)

    el.index_map((n_seg,), _check_mono, ordered=False)
  return offsets


def _record_ad_node(tape, el, kind: str, inputs: list, out: ArrayVal, info) -> None:
  from ..transforms.autodiff import Node

  node = Node(kind, tuple(inputs), info)
  tape.record(el, node)
  tape.register(el, node, out)


def _require_ragged(el, r, what: str) -> RaggedArray:
  if not isinstance(r, RaggedArray):
    raise el.error(TypeInferenceError, f"{what}() expects a RaggedArray, got {describe(r)}")
  return r


def _resolve_op(el, op, what: str) -> tuple[str | None, object]:
  if isinstance(op, str):
    key = _OP_ALIASES.get(op)
    if key is None:
      raise el.error(StaticValueError, f"{what}(): unknown op {op!r}")
    return key, prims.UFUNCS["add" if key == "mean" else key]
  if isinstance(op, prims.UFunc) and op.name in ("add", "mul", "min", "max"):
    return op.name, op
  return None, op


# ---------------------------------------------------------------------------
# segment_ids
# ---------------------------------------------------------------------------


@libfunction("segment_ids")
def segment_ids(el, offsets, total_len=None):
  """Return a 1D int64 array mapping each flat element index ``k`` to its segment ``b``."""
  if isinstance(offsets, RaggedArray):
    if total_len is None:
      total_len = _leading_dim(el, offsets.values)
    offsets = offsets.offsets
  offsets = el.as_array(offsets, "offsets")
  if total_len is not None:
    total_len = el.as_index(total_len, "total_len")
  offsets = check_ragged_offsets(el, offsets, total_len)
  n_seg = el.tmp(isub(offsets.shape[0], ONE), "nseg")
  if total_len is None:
    total_len = el.tmp(cast(el.elem(offsets, (n_seg,)), I), "total")
  out = el.alloc_array(Int64, (total_len,), zero=True, label="seg_ids")

  def fill_seg(idx):
    b = idx[0]
    s = el.tmp(cast(el.elem(offsets, (b,)), I), "s")
    e = el.tmp(cast(el.elem(offsets, (iadd(b, ONE),)), I), "e")
    k = Var("k", I)
    el.nonneg.add(k.uid)
    loop_scope = el.new_scope("loop")
    body: list = []
    with el.in_block(body, loop_scope, loop=True):
      el.emit(Store(out.buf, k, cast(b, Int64)))
    el.emit(ForRange(k, s, e, ONE, body, loop_scope))

  el.index_map((n_seg,), fill_seg, ordered=False)
  return out


# ---------------------------------------------------------------------------
# segmented_map
# ---------------------------------------------------------------------------


@libfunction("segmented_map")
def segmented_map(el, fn, ragged, *more):
  """Apply ``fn`` to every element of ``ragged`` (preserving CSR ``offsets`` zero-copy)."""
  r0 = _require_ragged(el, ragged, "segmented_map")
  offsets = check_ragged_offsets(el, r0.offsets, _leading_dim(el, r0.values))
  extra_vals = []
  for other in more:
    if isinstance(other, RaggedArray):
      o_off = check_ragged_offsets(el, other.offsets, _leading_dim(el, other.values))
      if el.options.bounds_check:
        el.check(
          cmp("eq", offsets.shape[0], o_off.shape[0]),
          "ValueError",
          "segmented_map() RaggedArray arguments have different segment counts ({0} vs {1})",
          isub(offsets.shape[0], ONE),
          isub(o_off.shape[0], ONE),
        )
      extra_vals.append(other.values)
    else:
      extra_vals.append(other)

  from .adverbs import map as _pk_map

  mapped = _pk_map.impl(el, fn, r0.values, *extra_vals, axis=0)
  return RaggedArray(mapped, offsets)


# ---------------------------------------------------------------------------
# segmented_reduce
# ---------------------------------------------------------------------------


def _neutral_init(op_name: str, elt):
  if op_name in ("add", "mean"):
    return Const(0.0 if elt.kind == "f" else 0, elt)
  if op_name == "mul":
    return Const(1.0 if elt.kind == "f" else 1, elt)
  if op_name == "max":
    if elt.kind == "f":
      return Const(-math.inf, elt)
    if elt.kind == "u":
      return Const(0, elt)
    return Const(-(1 << (elt.bits - 1)), elt)
  if op_name == "min":
    if elt.kind == "f":
      return Const(math.inf, elt)
    if elt.kind == "u":
      return Const((1 << elt.bits) - 1, elt)
    return Const((1 << (elt.bits - 1)) - 1, elt)
  return None


@libfunction("segmented_reduce")
def segmented_reduce(el, op, ragged, init=None):
  """Reduce each segment of ``ragged`` along its variable-length axis, returning one result per segment."""
  r = _require_ragged(el, ragged, "segmented_reduce")
  vals = r.values
  if not isinstance(vals, ArrayVal) or vals.rank < 1:
    raise el.error(TypeInferenceError, f"segmented_reduce() expects array values, got {describe(vals)}")
  vals = el.as_concrete(vals)
  offsets = check_ragged_offsets(el, r.offsets, vals.shape[0])
  n_seg = el.tmp(isub(offsets.shape[0], ONE), "nseg")
  op_name, op_fn = _resolve_op(el, op, "segmented_reduce")
  trailing = vals.shape[1:]
  out_elt = vals.elt
  if op_name == "mean" and out_elt.kind != "f":
    out_elt = Float64
  if init is not None:
    init_e = el.as_scalar(init, "init")
    from ..ndtypes import join

    out_elt = join(out_elt, init_e.type).strong
  else:
    init_e = None

  tape = el.tape
  record_ad = (
    tape is not None
    and not tape.muted
    and op_name in ("add", "mean", "max", "min")
    and out_elt.kind == "f"
    and tape.is_active(el, vals)
  )

  out_shape = (n_seg, *trailing)
  out = el.alloc_array(out_elt, out_shape, label="seg_red")

  def run_seg(idx):
    b = idx[0]
    t_idx = idx[1:]
    s = el.tmp(cast(el.elem(offsets, (b,)), I), "s")
    e = el.tmp(cast(el.elem(offsets, (iadd(b, ONE),)), I), "e")
    acc = Var("acc", out_elt)
    el.mutables.add(acc.uid)
    has_init = init_e is not None
    if has_init:
      el.emit(Assign(acc, cast(init_e, out_elt)))
      start_k = s
    else:
      el.check(cmp("lt", s, e), "ValueError", "segmented_reduce() on empty segment without init")
      first_v = cast(el.elem(vals, (s, *t_idx)), out_elt)
      el.emit(Assign(acc, first_v))
      start_k = iadd(s, ONE)
    k = Var("k", I)
    el.nonneg.add(k.uid)
    lscope = el.new_scope("loop")
    lbody: list = []
    with el.in_block(lbody, lscope, loop=True):
      xk = cast(el.elem(vals, (k, *t_idx)), out_elt)
      if op_name in ("add", "mean", "mul", "min", "max"):
        prim_op = "add" if op_name == "mean" else op_name
        nxt = ops.prim(prim_op, (acc, xk), out_elt)
      else:
        nxt = cast(el.as_scalar(el.call(op_fn, [acc, xk]), "segmented_reduce step"), out_elt)
      el.emit(Assign(acc, nxt))
    el.emit(ForRange(k, start_k, e, ONE, lbody, lscope))
    res = acc
    if op_name == "mean":
      cnt = el.tmp(isub(e, s), "cnt")
      denom = ops.select(cmp("gt", cnt, ZERO), cast(cnt, out_elt), Const(1.0, out_elt))
      res = ops.prim("div", (acc, denom), out_elt)
    out_pos = el.index_expr(out, idx)
    el.emit(Store(out.buf, out_pos, res))

  saved_tape = el.tape
  if record_ad:
    el.tape = None
  try:
    el.index_map(out_shape, run_seg, ordered=False)
  finally:
    el.tape = saved_tape

  if record_ad:
    _record_ad_node(tape, el, "segmented_reduce", [vals], out, (op_name, offsets, out, init_e))
  return out


def _segmented_reduce_vjp(tape, el, node, g: ArrayVal) -> None:
  (vals,) = node.inputs
  op_name, offsets, fwd_out, _init_e = node.info
  tape.stale(el, node, vals)
  g = el.as_concrete(g)
  n_seg = el.tmp(isub(offsets.shape[0], ONE), "nseg")
  trailing = vals.shape[1:]
  dvals = el.alloc_array(vals.elt, vals.shape, zero=True, label="d_seg_vals")

  def bwd_seg(idx):
    b = idx[0]
    t_idx = idx[1:]
    s = el.tmp(cast(el.elem(offsets, (b,)), I), "s")
    e = el.tmp(cast(el.elem(offsets, (iadd(b, ONE),)), I), "e")
    gb = el.tmp(cast(el.elem(g, idx), vals.elt), "gb")
    if op_name == "mean":
      cnt = el.tmp(isub(e, s), "cnt")
      denom = ops.select(cmp("gt", cnt, ZERO), cast(cnt, vals.elt), Const(1.0, vals.elt))
      gb = el.tmp(ops.prim("div", (gb, denom), vals.elt), "gb_mean")
    if op_name in ("add", "mean"):
      k = Var("k", I)
      el.nonneg.add(k.uid)
      lscope = el.new_scope("loop")
      lbody: list = []
      with el.in_block(lbody, lscope, loop=True):
        pos = el.index_expr(dvals, (k, *t_idx))
        el.emit(Store(dvals.buf, pos, gb))
      el.emit(ForRange(k, s, e, ONE, lbody, lscope))
    else:
      # max / min: split gradient equally among winning elements (matches NumPy/Parakeet reduce VJP)
      win = el.tmp(cast(el.elem(fwd_out, idx), vals.elt), "win")
      ties = Var("ties", I)
      el.mutables.add(ties.uid)
      el.emit(Assign(ties, ZERO))
      k1 = Var("k1", I)
      el.nonneg.add(k1.uid)
      sc1 = el.new_scope("loop")
      b1: list = []
      with el.in_block(b1, sc1, loop=True):
        xk = el.elem(vals, (k1, *t_idx))
        is_w = cmp("eq", xk, win)
        el.emit(Assign(ties, iadd(ties, ops.select(is_w, ONE, ZERO))))
      el.emit(ForRange(k1, s, e, ONE, b1, sc1))
      share = el.tmp(
        ops.prim("div", (gb, ops.select(cmp("gt", ties, ZERO), cast(ties, vals.elt), Const(1.0, vals.elt))), vals.elt),
        "share",
      )
      k2 = Var("k2", I)
      el.nonneg.add(k2.uid)
      sc2 = el.new_scope("loop")
      b2: list = []
      with el.in_block(b2, sc2, loop=True):
        xk = el.elem(vals, (k2, *t_idx))
        pos = el.index_expr(dvals, (k2, *t_idx))
        el.emit(Store(dvals.buf, pos, ops.select(cmp("eq", xk, win), share, Const(0.0, vals.elt))))
      el.emit(ForRange(k2, s, e, ONE, b2, sc2))

  el.index_map((n_seg, *trailing), bwd_seg, ordered=False)
  tape.add(el, vals, dvals)


# ---------------------------------------------------------------------------
# segmented_scan
# ---------------------------------------------------------------------------


@libfunction("segmented_scan")
def segmented_scan(el, op, ragged, *, exclusive=False, init=None):
  """Prefix scan reset at every segment boundary, returning a :class:`RaggedArray`."""
  r = _require_ragged(el, ragged, "segmented_scan")
  vals = r.values
  if not isinstance(vals, ArrayVal) or vals.rank < 1:
    raise el.error(TypeInferenceError, f"segmented_scan() expects array values, got {describe(vals)}")
  vals = el.as_concrete(vals)
  offsets = check_ragged_offsets(el, r.offsets, vals.shape[0])
  excl = el.static_bool(exclusive, "exclusive")
  n_seg = el.tmp(isub(offsets.shape[0], ONE), "nseg")
  op_name, op_fn = _resolve_op(el, op, "segmented_scan")
  if op_name == "mean":
    raise el.error(StaticValueError, "segmented_scan() does not support 'mean'")
  out_elt = vals.elt
  if init is not None:
    init_e = el.as_scalar(init, "init")
    from ..ndtypes import join

    out_elt = join(out_elt, init_e.type).strong
  elif excl:
    init_e = _neutral_init(op_name or "", out_elt)
    if init_e is None:
      raise el.error(StaticValueError, "exclusive segmented_scan() with a custom op requires init=")
  else:
    init_e = None

  trailing = vals.shape[1:]
  out = el.alloc_array(out_elt, vals.shape, label="seg_scan")

  def run_seg(idx):
    b = idx[0]
    t_idx = idx[1:]
    s = el.tmp(cast(el.elem(offsets, (b,)), I), "s")
    e = el.tmp(cast(el.elem(offsets, (iadd(b, ONE),)), I), "e")
    acc = Var("acc", out_elt)
    el.mutables.add(acc.uid)
    k = Var("k", I)
    el.nonneg.add(k.uid)
    lscope = el.new_scope("loop")
    lbody: list = []

    def step(a, x):
      if op_name in ("add", "mul", "min", "max"):
        return ops.prim(op_name, (a, x), out_elt)
      return cast(el.as_scalar(el.call(op_fn, [a, x]), "segmented_scan step"), out_elt)

    if excl:
      el.emit(Assign(acc, cast(init_e, out_elt)))
      with el.in_block(lbody, lscope, loop=True):
        xk = cast(el.elem(vals, (k, *t_idx)), out_elt)
        pos = el.index_expr(out, (k, *t_idx))
        el.emit(Store(out.buf, pos, acc))
        el.emit(Assign(acc, step(acc, xk)))
      el.emit(ForRange(k, s, e, ONE, lbody, lscope))
    else:
      non_empty = cmp("lt", s, e)
      then_blk: list = []
      with el.in_block(then_blk):
        first = cast(el.elem(vals, (s, *t_idx)), out_elt)
        if init_e is not None:
          first = step(cast(init_e, out_elt), first)
        el.emit(Assign(acc, first))
        el.emit(Store(out.buf, el.index_expr(out, (s, *t_idx)), acc))
        with el.in_block(lbody, lscope, loop=True):
          xk = cast(el.elem(vals, (k, *t_idx)), out_elt)
          el.emit(Assign(acc, step(acc, xk)))
          el.emit(Store(out.buf, el.index_expr(out, (k, *t_idx)), acc))
        el.emit(ForRange(k, iadd(s, ONE), e, ONE, lbody, lscope))
      el.emit(If(non_empty, then_blk, []))

  el.index_map((n_seg, *trailing), run_seg, ordered=False)
  return RaggedArray(out, offsets)


# ---------------------------------------------------------------------------
# segmented_softmax
# ---------------------------------------------------------------------------


@libfunction("segmented_softmax")
def segmented_softmax(el, ragged):
  """Numerically stable per-segment softmax over a :class:`RaggedArray`."""
  r = _require_ragged(el, ragged, "segmented_softmax")
  vals = r.values
  if not isinstance(vals, ArrayVal) or vals.rank < 1:
    raise el.error(TypeInferenceError, f"segmented_softmax() expects array values, got {describe(vals)}")
  if vals.elt.kind != "f":
    vals = el.cast_array(vals, Float32)
  vals = el.as_concrete(vals)
  offsets = check_ragged_offsets(el, r.offsets, vals.shape[0])
  n_seg = el.tmp(isub(offsets.shape[0], ONE), "nseg")
  elt = vals.elt
  trailing = vals.shape[1:]

  tape = el.tape
  record_ad = tape is not None and not tape.muted and tape.is_active(el, vals)
  out = el.alloc_array(elt, vals.shape, label="seg_sm")

  def run_seg(idx):
    b = idx[0]
    t_idx = idx[1:]
    s = el.tmp(cast(el.elem(offsets, (b,)), I), "s")
    e = el.tmp(cast(el.elem(offsets, (iadd(b, ONE),)), I), "e")
    then_blk: list = []
    with el.in_block(then_blk):
      # Online (m, l) pass
      m = Var("m", elt)
      l_sum = Var("l", elt)
      el.mutables.update((m.uid, l_sum.uid))
      el.emit(Assign(m, Const(-math.inf, elt)))
      el.emit(Assign(l_sum, Const(0.0, elt)))
      k1 = Var("k1", I)
      el.nonneg.add(k1.uid)
      sc1 = el.new_scope("loop")
      b1: list = []
      with el.in_block(b1, sc1, loop=True):
        xk = el.tmp(el.elem(vals, (k1, *t_idx)), "x")
        m_new = el.tmp(ops.prim("max", (m, xk), elt), "mn")
        scale_old = ops.prim("exp", (ops.prim("sub", (m, m_new), elt),), elt)
        term = ops.prim("exp", (ops.prim("sub", (xk, m_new), elt),), elt)
        el.emit(Assign(l_sum, ops.prim("add", (ops.prim("mul", (l_sum, scale_old), elt), term), elt)))
        el.emit(Assign(m, m_new))
      el.emit(ForRange(k1, s, e, ONE, b1, sc1))
      inv_l = el.tmp(ops.prim("div", (Const(1.0, elt), l_sum), elt), "inv_l")
      k2 = Var("k2", I)
      el.nonneg.add(k2.uid)
      sc2 = el.new_scope("loop")
      b2: list = []
      with el.in_block(b2, sc2, loop=True):
        xk = el.elem(vals, (k2, *t_idx))
        p = ops.prim("mul", (ops.prim("exp", (ops.prim("sub", (xk, m), elt),), elt), inv_l), elt)
        el.emit(Store(out.buf, el.index_expr(out, (k2, *t_idx)), p))
      el.emit(ForRange(k2, s, e, ONE, b2, sc2))
    el.emit(If(cmp("lt", s, e), then_blk, []))

  saved_tape = el.tape
  if record_ad:
    el.tape = None
  try:
    el.index_map((n_seg, *trailing), run_seg, ordered=False)
  finally:
    el.tape = saved_tape

  if record_ad:
    _record_ad_node(tape, el, "segmented_softmax", [vals], out, (offsets, out))
  return RaggedArray(out, offsets)


def _segmented_softmax_vjp(tape, el, node, g: ArrayVal) -> None:
  (vals,) = node.inputs
  offsets, p_arr = node.info
  tape.stale(el, node, vals)
  g = el.as_concrete(g)
  n_seg = el.tmp(isub(offsets.shape[0], ONE), "nseg")
  elt = vals.elt
  trailing = vals.shape[1:]
  dvals = el.alloc_array(elt, vals.shape, zero=True, label="d_seg_sm")

  def bwd_seg(idx):
    b = idx[0]
    t_idx = idx[1:]
    s = el.tmp(cast(el.elem(offsets, (b,)), I), "s")
    e = el.tmp(cast(el.elem(offsets, (iadd(b, ONE),)), I), "e")
    dot = Var("dot", elt)
    el.mutables.add(dot.uid)
    el.emit(Assign(dot, Const(0.0, elt)))
    k1 = Var("k1", I)
    el.nonneg.add(k1.uid)
    sc1 = el.new_scope("loop")
    b1: list = []
    with el.in_block(b1, sc1, loop=True):
      pk = el.elem(p_arr, (k1, *t_idx))
      gk = cast(el.elem(g, (k1, *t_idx)), elt)
      el.emit(Assign(dot, ops.prim("add", (dot, ops.prim("mul", (pk, gk), elt)), elt)))
    el.emit(ForRange(k1, s, e, ONE, b1, sc1))
    k2 = Var("k2", I)
    el.nonneg.add(k2.uid)
    sc2 = el.new_scope("loop")
    b2: list = []
    with el.in_block(b2, sc2, loop=True):
      pk = el.elem(p_arr, (k2, *t_idx))
      gk = cast(el.elem(g, (k2, *t_idx)), elt)
      dx = ops.prim("mul", (pk, ops.prim("sub", (gk, dot), elt)), elt)
      el.emit(Store(dvals.buf, el.index_expr(dvals, (k2, *t_idx)), dx))
    el.emit(ForRange(k2, s, e, ONE, b2, sc2))

  el.index_map((n_seg, *trailing), bwd_seg, ordered=False)
  tape.add(el, vals, dvals)


# ---------------------------------------------------------------------------
# Varlen FlashAttention (Law 1 online monoid + Law 2 D_i-recomputed backward)
# ---------------------------------------------------------------------------


def _unpack_qkv_3d(el, x, what: str) -> tuple[ArrayVal, bool]:
  if not isinstance(x, ArrayVal) or x.rank not in (2, 3):
    raise el.error(TypeInferenceError, f"{what} must be 2D (total, D) or 3D (total, H, D), got {describe(x)}")
  if x.elt.kind != "f":
    raise el.error(TypeInferenceError, f"{what} must be floating-point, got {describe(x)}")
  if x.rank == 2:
    x = el.reshape(x, (x.shape[0], ONE, x.shape[1]), checked=False)
    return el.as_concrete(x), True
  return el.as_concrete(x), False


@libfunction("varlen_flash_attention")
def varlen_flash_attention(el, q, k, v, cu_seqlens=None, *, cu_seqlens_kv=None, scale=None, causal=False, window=None):
  """Packed variable-length FlashAttention with zero padding waste and ``O(total * H * D)`` memory.

  Accepts either packed tensors ``(q, k, v, cu_seqlens)`` of shape ``(total, H, D)``
  (or ``(total, D)``) or :class:`~parakeet.RaggedArray` arguments ``(q, k, v)``.
  Supports optional ``causal=True``, sliding ``window=W``, and reverse-mode
  differentiation via Law 2 ``D_i``-recomputed backward (saving only ``LSE``).
  """
  return_ragged = isinstance(q, RaggedArray)
  if isinstance(q, RaggedArray):
    if cu_seqlens is None:
      cu_seqlens = q.offsets
    q_arr = q.values
  else:
    q_arr = q
  if isinstance(k, RaggedArray):
    if cu_seqlens_kv is None:
      cu_seqlens_kv = k.offsets
    k_arr = k.values
  else:
    k_arr = k
  if isinstance(v, RaggedArray):
    if cu_seqlens_kv is None:
      cu_seqlens_kv = v.offsets
    v_arr = v.values
  else:
    v_arr = v

  if cu_seqlens is None:
    raise el.error(TypeInferenceError, "varlen_flash_attention() requires cu_seqlens or RaggedArray inputs")
  if cu_seqlens_kv is None:
    cu_seqlens_kv = cu_seqlens

  q3, was_2d = _unpack_qkv_3d(el, q_arr, "q")
  k3, k_2d = _unpack_qkv_3d(el, k_arr, "k")
  v3, v_2d = _unpack_qkv_3d(el, v_arr, "v")
  if not (was_2d == k_2d == v_2d):
    raise el.error(TypeInferenceError, "q, k, v must have the same rank (all 2D or all 3D)")

  elt = q3.elt
  if k3.elt is not elt:
    k3 = el.as_concrete(el.cast_array(k3, elt))
  if v3.elt is not elt:
    v3 = el.as_concrete(el.cast_array(v3, elt))

  total_q, H, K_dim = q3.shape
  total_kv, H_kv, K_dim_k = k3.shape
  total_v, H_v, D_dim = v3.shape
  if el.options.bounds_check:
    el.check(cmp("eq", H, H_kv), "ValueError", "varlen_flash_attention: head count mismatch ({0} vs {1})", H, H_kv)
    el.check(cmp("eq", H, H_v), "ValueError", "varlen_flash_attention: head count mismatch ({0} vs {1})", H, H_v)
    el.check(
      cmp("eq", K_dim, K_dim_k), "ValueError", "varlen_flash_attention: K dim mismatch ({0} vs {1})", K_dim, K_dim_k
    )
    el.check(
      cmp("eq", total_kv, total_v),
      "ValueError",
      "varlen_flash_attention: k/v length mismatch ({0} vs {1})",
      total_kv,
      total_v,
    )

  off_q = check_ragged_offsets(el, el.as_array(cu_seqlens, "cu_seqlens"), total_q)
  off_kv = (
    off_q
    if cu_seqlens_kv is cu_seqlens
    else check_ragged_offsets(el, el.as_array(cu_seqlens_kv, "cu_seqlens_kv"), total_kv)
  )
  if el.options.bounds_check and off_kv is not off_q:
    el.check(
      cmp("eq", off_q.shape[0], off_kv.shape[0]),
      "ValueError",
      "varlen_flash_attention: cu_seqlens and cu_seqlens_kv have different batch sizes",
    )

  is_causal = el.static_bool(causal, "causal")
  win = None if window is None else el.as_index(window, "window")
  if scale is None:
    sc = el.tmp(ops.prim("div", (Const(1.0, elt), ops.prim("sqrt", (cast(K_dim, elt),), elt)), elt), "scale")
  else:
    sc = el.tmp(cast(el.as_scalar(scale, "scale"), elt), "scale")

  tape = el.tape
  record_ad = (
    tape is not None and not tape.muted and (tape.is_active(el, q3) or tape.is_active(el, k3) or tape.is_active(el, v3))
  )

  saved_tape = el.tape
  if record_ad:
    el.tape = None
  try:
    q_seg = segment_ids.impl(el, off_q, total_q)
    out = el.alloc_array(elt, (total_q, H, D_dim), zero=True, label="vfa_out")
    lse = el.alloc_array(elt, (total_q, H), zero=True, label="vfa_lse") if record_ad else None

    def fwd_token(idx):
      qi, h = idx
      b = el.tmp(cast(el.elem(q_seg, (qi,)), I), "b")
      qs = el.tmp(cast(el.elem(off_q, (b,)), I), "qs")
      ks = el.tmp(cast(el.elem(off_kv, (b,)), I), "ks")
      ke = el.tmp(cast(el.elem(off_kv, (iadd(b, ONE),)), I), "ke")
      qloc = el.tmp(isub(qi, qs), "qloc")
      lo = ks
      hi = ke
      if is_causal:
        hi = el.tmp(imin(hi, iadd(iadd(ks, qloc), ONE)), "hi")
      if win is not None:
        lo = el.tmp(imax(lo, isub(iadd(ks, qloc), win)), "lo")
        if not is_causal:
          hi = el.tmp(imin(hi, iadd(iadd(iadd(ks, qloc), win), ONE)), "hi")

      m = Var("m", elt)
      l_acc = Var("l", elt)
      el.mutables.update((m.uid, l_acc.uid))
      el.emit(Assign(m, Const(-math.inf, elt)))
      el.emit(Assign(l_acc, Const(0.0, elt)))

      # Online 1-pass (m, l, O) monoid over kj in [lo, hi)
      kj = Var("kj", I)
      el.nonneg.add(kj.uid)
      kscope = el.new_scope("loop")
      kbody: list = []
      with el.in_block(kbody, kscope, loop=True):
        # dot = sum_c Q[qi, h, c] * K[kj, h, c]
        dot = Var("dot", elt)
        el.mutables.add(dot.uid)
        el.emit(Assign(dot, Const(0.0, elt)))
        c = Var("c", I)
        el.nonneg.add(c.uid)
        cscope = el.new_scope("loop")
        cbody: list = []
        with el.in_block(cbody, cscope, loop=True):
          qv = el.elem(q3, (qi, h, c))
          kv = el.elem(k3, (kj, h, c))
          el.emit(Assign(dot, ops.prim("add", (dot, ops.prim("mul", (qv, kv), elt)), elt)))
        el.emit(ForRange(c, ZERO, K_dim, ONE, cbody, cscope))
        score = el.tmp(ops.prim("mul", (dot, sc), elt), "score")
        m_new = el.tmp(ops.prim("max", (m, score), elt), "mn")
        alpha = el.tmp(ops.prim("exp", (ops.prim("sub", (m, m_new), elt),), elt), "alpha")
        beta = el.tmp(ops.prim("exp", (ops.prim("sub", (score, m_new), elt),), elt), "beta")
        el.emit(Assign(l_acc, ops.prim("add", (ops.prim("mul", (l_acc, alpha), elt), beta), elt)))
        el.emit(Assign(m, m_new))
        # Rescale running O[qi, h, d] and accumulate beta * V[kj, h, d]
        d = Var("d", I)
        el.nonneg.add(d.uid)
        dscope = el.new_scope("loop")
        dbody: list = []
        with el.in_block(dbody, dscope, loop=True):
          opos = el.index_expr(out, (qi, h, d))
          cur = Load(out.buf, opos)
          vv = el.elem(v3, (kj, h, d))
          upd = ops.prim("add", (ops.prim("mul", (cur, alpha), elt), ops.prim("mul", (beta, vv), elt)), elt)
          el.emit(Store(out.buf, opos, upd))
        el.emit(ForRange(d, ZERO, D_dim, ONE, dbody, dscope))
      el.emit(ForRange(kj, lo, hi, ONE, kbody, kscope))

      # Normalize O[qi, h, :] /= l_acc (safe for empty [lo, hi))
      has_any = cmp("gt", l_acc, Const(0.0, elt))
      inv_l = el.tmp(ops.select(has_any, ops.prim("div", (Const(1.0, elt), l_acc), elt), Const(0.0, elt)), "inv_l")
      d2 = Var("d2", I)
      el.nonneg.add(d2.uid)
      d2scope = el.new_scope("loop")
      d2body: list = []
      with el.in_block(d2body, d2scope, loop=True):
        opos = el.index_expr(out, (qi, h, d2))
        el.emit(Store(out.buf, opos, ops.prim("mul", (Load(out.buf, opos), inv_l), elt)))
      el.emit(ForRange(d2, ZERO, D_dim, ONE, d2body, d2scope))
      if lse is not None:
        lse_val = ops.select(has_any, ops.prim("add", (m, ops.prim("log", (l_acc,), elt)), elt), Const(-math.inf, elt))
        el.emit(Store(lse.buf, el.index_expr(lse, (qi, h)), lse_val))

    el.index_map((total_q, H), fwd_token, ordered=False)
  finally:
    el.tape = saved_tape

  if record_ad:
    _record_ad_node(
      tape,
      el,
      "varlen_flash_attention",
      [q3, k3, v3],
      out,
      (off_q, off_kv, q_seg, out, lse, sc, is_causal, win),
    )

  res_arr = el.reshape(out, (total_q, D_dim), checked=False) if was_2d else out
  return RaggedArray(res_arr, off_q) if return_ragged else res_arr


def _varlen_flash_attention_vjp(tape, el, node, g: ArrayVal) -> None:
  """Law 2 ``D_i``-recomputed varlen FlashAttention backward: zero ``T^2`` memory, zero atomics."""
  q3, k3, v3 = node.inputs
  off_q, off_kv, q_seg, fwd_out, lse, sc, is_causal, win = node.info
  tape.stale(el, node, q3)
  tape.stale(el, node, k3)
  tape.stale(el, node, v3)
  g = el.as_concrete(el.cast_array(g, q3.elt))
  elt = q3.elt
  total_q, H, K_dim = q3.shape
  total_kv, _, D_dim = v3.shape

  need_q = tape.is_active(el, q3)
  need_k = tape.is_active(el, k3)
  need_v = tape.is_active(el, v3)

  # Step 1: D[qi, h] = sum_d dO[qi, h, d] * O[qi, h, d]
  d_vec = el.alloc_array(elt, (total_q, H), label="vfa_D")

  def compute_d(idx):
    qi, h = idx
    acc = Var("d_acc", elt)
    el.mutables.add(acc.uid)
    el.emit(Assign(acc, Const(0.0, elt)))
    d = Var("d", I)
    el.nonneg.add(d.uid)
    dsc = el.new_scope("loop")
    db: list = []
    with el.in_block(db, dsc, loop=True):
      el.emit(
        Assign(
          acc, ops.prim("add", (acc, ops.prim("mul", (el.elem(g, (qi, h, d)), el.elem(fwd_out, (qi, h, d))), elt)), elt)
        )
      )
    el.emit(ForRange(d, ZERO, D_dim, ONE, db, dsc))
    el.emit(Store(d_vec.buf, el.index_expr(d_vec, (qi, h)), acc))

  el.index_map((total_q, H), compute_d, ordered=False)

  # Step 2: Atomic-free dQ parallel over (total_q, H)
  if need_q:
    dq = el.alloc_array(elt, q3.shape, zero=True, label="vfa_dQ")

    def compute_dq(idx):
      qi, h = idx
      b = el.tmp(cast(el.elem(q_seg, (qi,)), I), "b")
      qs = el.tmp(cast(el.elem(off_q, (b,)), I), "qs")
      ks = el.tmp(cast(el.elem(off_kv, (b,)), I), "ks")
      ke = el.tmp(cast(el.elem(off_kv, (iadd(b, ONE),)), I), "ke")
      qloc = el.tmp(isub(qi, qs), "qloc")
      lo, hi = ks, ke
      if is_causal:
        hi = el.tmp(imin(hi, iadd(iadd(ks, qloc), ONE)), "hi")
      if win is not None:
        lo = el.tmp(imax(lo, isub(iadd(ks, qloc), win)), "lo")
        if not is_causal:
          hi = el.tmp(imin(hi, iadd(iadd(iadd(ks, qloc), win), ONE)), "hi")
      lse_i = el.tmp(el.elem(lse, (qi, h)), "lse_i")
      d_i = el.tmp(el.elem(d_vec, (qi, h)), "d_i")

      kj = Var("kj", I)
      el.nonneg.add(kj.uid)
      ksc = el.new_scope("loop")
      kb: list = []
      with el.in_block(kb, ksc, loop=True):
        dot = Var("dot", elt)
        el.mutables.add(dot.uid)
        el.emit(Assign(dot, Const(0.0, elt)))
        c = Var("c", I)
        el.nonneg.add(c.uid)
        csc = el.new_scope("loop")
        cb: list = []
        with el.in_block(cb, csc, loop=True):
          el.emit(
            Assign(
              dot, ops.prim("add", (dot, ops.prim("mul", (el.elem(q3, (qi, h, c)), el.elem(k3, (kj, h, c))), elt)), elt)
            )
          )
        el.emit(ForRange(c, ZERO, K_dim, ONE, cb, csc))
        p_ij = el.tmp(ops.prim("exp", (ops.prim("sub", (ops.prim("mul", (dot, sc), elt), lse_i), elt),), elt), "p")
        dp = Var("dp", elt)
        el.mutables.add(dp.uid)
        el.emit(Assign(dp, Const(0.0, elt)))
        d = Var("d", I)
        el.nonneg.add(d.uid)
        dsc = el.new_scope("loop")
        db: list = []
        with el.in_block(db, dsc, loop=True):
          el.emit(
            Assign(
              dp, ops.prim("add", (dp, ops.prim("mul", (el.elem(g, (qi, h, d)), el.elem(v3, (kj, h, d))), elt)), elt)
            )
          )
        el.emit(ForRange(d, ZERO, D_dim, ONE, db, dsc))
        ds = el.tmp(ops.prim("mul", (ops.prim("mul", (p_ij, ops.prim("sub", (dp, d_i), elt)), elt), sc), elt), "ds")
        c2 = Var("c2", I)
        el.nonneg.add(c2.uid)
        c2sc = el.new_scope("loop")
        c2b: list = []
        with el.in_block(c2b, c2sc, loop=True):
          qpos = el.index_expr(dq, (qi, h, c2))
          el.emit(
            Store(
              dq.buf,
              qpos,
              ops.prim("add", (Load(dq.buf, qpos), ops.prim("mul", (ds, el.elem(k3, (kj, h, c2))), elt)), elt),
            )
          )
        el.emit(ForRange(c2, ZERO, K_dim, ONE, c2b, c2sc))
      el.emit(ForRange(kj, lo, hi, ONE, kb, ksc))

    el.index_map((total_q, H), compute_dq, ordered=False)
    tape.add(el, q3, dq)

  # Step 3: Atomic-free dK & dV parallel over (total_kv, H)
  if need_k or need_v:
    kv_seg = q_seg if off_kv is off_q else segment_ids.impl(el, off_kv, total_kv)
    dk = el.alloc_array(elt, k3.shape, zero=True, label="vfa_dK") if need_k else None
    dv = el.alloc_array(elt, v3.shape, zero=True, label="vfa_dV") if need_v else None

    def compute_dkv(idx):
      kj, h = idx
      b = el.tmp(cast(el.elem(kv_seg, (kj,)), I), "b")
      qs = el.tmp(cast(el.elem(off_q, (b,)), I), "qs")
      qe = el.tmp(cast(el.elem(off_q, (iadd(b, ONE),)), I), "qe")
      ks = el.tmp(cast(el.elem(off_kv, (b,)), I), "ks")
      kloc = el.tmp(isub(kj, ks), "kloc")
      # Invert causal & window bounds: qi_local >= kloc (if causal), kloc <= qi_local <= kloc + win (if causal+win), etc.
      qlo, qhi = qs, qe
      if is_causal:
        qlo = el.tmp(imax(qlo, iadd(qs, kloc)), "qlo")
      if win is not None:
        if not is_causal:
          qlo = el.tmp(imax(qlo, isub(iadd(qs, kloc), win)), "qlo")
        qhi = el.tmp(imin(qhi, iadd(iadd(iadd(qs, kloc), win), ONE)), "qhi")

      qi = Var("qi", I)
      el.nonneg.add(qi.uid)
      qsc = el.new_scope("loop")
      qb: list = []
      with el.in_block(qb, qsc, loop=True):
        dot = Var("dot", elt)
        el.mutables.add(dot.uid)
        el.emit(Assign(dot, Const(0.0, elt)))
        c = Var("c", I)
        el.nonneg.add(c.uid)
        csc = el.new_scope("loop")
        cb: list = []
        with el.in_block(cb, csc, loop=True):
          el.emit(
            Assign(
              dot, ops.prim("add", (dot, ops.prim("mul", (el.elem(q3, (qi, h, c)), el.elem(k3, (kj, h, c))), elt)), elt)
            )
          )
        el.emit(ForRange(c, ZERO, K_dim, ONE, cb, csc))
        lse_i = el.elem(lse, (qi, h))
        p_ij = el.tmp(ops.prim("exp", (ops.prim("sub", (ops.prim("mul", (dot, sc), elt), lse_i), elt),), elt), "p")
        if dv is not None:
          d = Var("d", I)
          el.nonneg.add(d.uid)
          dsc = el.new_scope("loop")
          db: list = []
          with el.in_block(db, dsc, loop=True):
            vpos = el.index_expr(dv, (kj, h, d))
            el.emit(
              Store(
                dv.buf,
                vpos,
                ops.prim("add", (Load(dv.buf, vpos), ops.prim("mul", (p_ij, el.elem(g, (qi, h, d))), elt)), elt),
              )
            )
          el.emit(ForRange(d, ZERO, D_dim, ONE, db, dsc))
        if dk is not None:
          dp = Var("dp", elt)
          el.mutables.add(dp.uid)
          el.emit(Assign(dp, Const(0.0, elt)))
          d2 = Var("d2", I)
          el.nonneg.add(d2.uid)
          d2sc = el.new_scope("loop")
          d2b: list = []
          with el.in_block(d2b, d2sc, loop=True):
            el.emit(
              Assign(
                dp,
                ops.prim("add", (dp, ops.prim("mul", (el.elem(g, (qi, h, d2)), el.elem(v3, (kj, h, d2))), elt)), elt),
              )
            )
          el.emit(ForRange(d2, ZERO, D_dim, ONE, d2b, d2sc))
          d_i = el.elem(d_vec, (qi, h))
          ds = el.tmp(ops.prim("mul", (ops.prim("mul", (p_ij, ops.prim("sub", (dp, d_i), elt)), elt), sc), elt), "ds")
          c2 = Var("c2", I)
          el.nonneg.add(c2.uid)
          c2sc = el.new_scope("loop")
          c2b: list = []
          with el.in_block(c2b, c2sc, loop=True):
            kpos = el.index_expr(dk, (kj, h, c2))
            el.emit(
              Store(
                dk.buf,
                kpos,
                ops.prim("add", (Load(dk.buf, kpos), ops.prim("mul", (ds, el.elem(q3, (qi, h, c2))), elt)), elt),
              )
            )
          el.emit(ForRange(c2, ZERO, K_dim, ONE, c2b, c2sc))
      el.emit(ForRange(qi, qlo, qhi, ONE, qb, qsc))

    el.index_map((total_kv, H), compute_dkv, ordered=False)
    if dk is not None:
      tape.add(el, k3, dk)
    if dv is not None:
      tape.add(el, v3, dv)
