"""Law 5 generalized: 3D Morton ordering, 3D AABB tile pruning, and CSR BlockMask sparse FlashAttention."""

from __future__ import annotations

import math

from ..errors import TypeInferenceError, UnsupportedError
from ..ndtypes import Bool, UInt64
from ..spatial import BlockMask
from ..syntax import ops
from ..syntax.ir import Assign, Const, Expr, ForRange, If, Load, Store, Var
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imul
from ..syntax.values import ArrayVal, describe
from .adverbs import libfunction

# ---------------------------------------------------------------------------
# 1. 3D Morton (Z-order) spatial ordering
# ---------------------------------------------------------------------------


def _spread_bits_20(v: Expr) -> Expr:
  """Spread the lower 20 bits of ``v`` (uint64) so adjacent bits sit 3 positions apart."""
  u = ops.prim("bitand", (cast(v, UInt64), Const(0xFFFFF, UInt64)), UInt64)
  u = ops.prim(
    "bitand",
    (
      ops.prim("bitor", (u, ops.prim("lshift", (u, Const(32, UInt64)), UInt64)), UInt64),
      Const(0x001F00000000FFFF, UInt64),
    ),
    UInt64,
  )
  u = ops.prim(
    "bitand",
    (
      ops.prim("bitor", (u, ops.prim("lshift", (u, Const(16, UInt64)), UInt64)), UInt64),
      Const(0x001F0000FF0000FF, UInt64),
    ),
    UInt64,
  )
  u = ops.prim(
    "bitand",
    (
      ops.prim("bitor", (u, ops.prim("lshift", (u, Const(8, UInt64)), UInt64)), UInt64),
      Const(0x100F00F00F00F00F, UInt64),
    ),
    UInt64,
  )
  u = ops.prim(
    "bitand",
    (
      ops.prim("bitor", (u, ops.prim("lshift", (u, Const(4, UInt64)), UInt64)), UInt64),
      Const(0x10C30C30C30C30C3, UInt64),
    ),
    UInt64,
  )
  u = ops.prim(
    "bitand",
    (
      ops.prim("bitor", (u, ops.prim("lshift", (u, Const(2, UInt64)), UInt64)), UInt64),
      Const(0x1249249249249249, UInt64),
    ),
    UInt64,
  )
  return u


@libfunction("morton_order")
def morton_order(el, coords):
  """Return permutation indices ordering 3D ``coords`` ``(N, 3)`` along a 60-bit Morton Z-order curve."""
  c = el.as_concrete(el.as_array(coords, "coords"))
  if c.rank != 2 or c.elt.kind != "f":
    raise el.error(TypeInferenceError, f"morton_order() expects a 2D float array of shape (N, 3), got {describe(c)}")
  n = c.shape[0]
  if el.options.bounds_check:
    el.check(cmp("eq", c.shape[1], Const(3, I)), "ValueError", "morton_order() requires 3D coordinates (N, 3)")
  elt = c.elt
  # Compute finite bounding box per axis in O(N)
  from .numpy_lib import amax, amin

  lo = el.alloc_array(elt, (Const(3, I),), zero=True, label="m_lo")
  hi = el.alloc_array(elt, (Const(3, I),), zero=True, label="m_hi")
  zero_f = Const(0.0, elt)
  pos_inf = Const(math.inf, elt)
  neg_inf = Const(-math.inf, elt)
  bbox_blk: list = []
  with el.in_block(bbox_blk):
    for axis_k in range(3):
      col = el.getitem(c, (slice(None, None, None), Const(axis_k, I)))
      # Replace non-finite values with +inf for min and -inf for max so NaNs never skew finite bounds
      safe_min = el.delayed(
        elt,
        (n,),
        lambda idx, _col=col: ops.select(
          ops.prim("isfinite", (el.elem(_col, idx),), Bool), el.elem(_col, idx), pos_inf
        ),
        cost=2,
      )
      safe_max = el.delayed(
        elt,
        (n,),
        lambda idx, _col=col: ops.select(
          ops.prim("isfinite", (el.elem(_col, idx),), Bool), el.elem(_col, idx), neg_inf
        ),
        cost=2,
      )
      mn = el.tmp(el.as_scalar(amin(el, safe_min, None, False)), "mn")
      mx = el.tmp(el.as_scalar(amax(el, safe_max, None, False)), "mx")
      valid = ops.land(ops.prim("isfinite", (mn,), Bool), ops.prim("isfinite", (mx,), Bool))
      el.emit(Store(lo.buf, Const(axis_k, I), ops.select(valid, mn, zero_f)))
      el.emit(Store(hi.buf, Const(axis_k, I), ops.select(valid, mx, zero_f)))
  el.emit(If(cmp("gt", n, ZERO), bbox_blk, []))

  scale_max = Const(1048575.0, elt)  # 2^20 - 1
  eps = Const(1e-12, elt)
  codes = el.alloc_array(UInt64, (n,), label="morton_code")

  def encode_i(idx):
    i = idx[0]
    qs = []
    for k in range(3):
      ck = el.tmp(el.elem(c, (i, Const(k, I))), "ck")
      mn = Load(lo.buf, Const(k, I))
      mx = Load(hi.buf, Const(k, I))
      span = ops.prim("max", (ops.prim("sub", (mx, mn), elt), eps), elt)
      norm = ops.prim("div", (ops.prim("sub", (ck, mn), elt), span), elt)
      clamped = ops.prim("min", (ops.prim("max", (norm, Const(0.0, elt)), elt), Const(1.0, elt)), elt)
      safe = ops.select(ops.prim("isfinite", (ck,), Bool), clamped, Const(0.0, elt))
      q_int = cast(ops.prim("mul", (safe, scale_max), elt), UInt64)
      qs.append(_spread_bits_20(q_int))
    code = ops.prim(
      "bitor",
      (
        ops.prim("bitor", (qs[0], ops.prim("lshift", (qs[1], Const(1, UInt64)), UInt64)), UInt64),
        ops.prim("lshift", (qs[2], Const(2, UInt64)), UInt64),
      ),
      UInt64,
    )
    el.emit(Store(codes.buf, i, code))

  el.index_map((n,), encode_i, ordered=False)
  from .numpy_lib import argsort

  return argsort(el, codes, axis=-1)


# ---------------------------------------------------------------------------
# 2. Per-tile 3D AABB builder & AABB-pruned spatial cutoff operators
# ---------------------------------------------------------------------------


def _build_tile_aabbs(el, coords: ArrayVal, tile_size: int) -> tuple[ArrayVal, ArrayVal, ArrayVal, Expr]:
  """Build per-tile 3D AABBs ``(mins[T, 3], maxs[T, 3], has_nan[T], num_tiles)`` in O(N)."""
  n = coords.shape[0]
  elt = coords.elt
  bs = Const(tile_size, I)
  n_tiles = el.tmp(idiv(iadd(n, Const(tile_size - 1, I)), bs), "ntiles")
  mins = el.alloc_array(elt, (n_tiles, Const(3, I)), label="aabb_min")
  maxs = el.alloc_array(elt, (n_tiles, Const(3, I)), label="aabb_max")
  has_nan = el.alloc_array(Bool, (n_tiles,), label="aabb_nan")

  def build_t(idx):
    t = idx[0]
    t0 = el.tmp(imul(t, bs), "t0")
    t1 = el.tmp(imin(iadd(t0, bs), n), "t1")
    mn_x = Var("mnx", elt)
    mn_y = Var("mny", elt)
    mn_z = Var("mnz", elt)
    mx_x = Var("mxx", elt)
    mx_y = Var("mxy", elt)
    mx_z = Var("mxz", elt)
    nan_f = Var("nanf", Bool)
    for v in (mn_x, mn_y, mn_z, mx_x, mx_y, mx_z, nan_f):
      el.mutables.add(v.uid)
    el.emit(Assign(mn_x, Const(math.inf, elt)))
    el.emit(Assign(mn_y, Const(math.inf, elt)))
    el.emit(Assign(mn_z, Const(math.inf, elt)))
    el.emit(Assign(mx_x, Const(-math.inf, elt)))
    el.emit(Assign(mx_y, Const(-math.inf, elt)))
    el.emit(Assign(mx_z, Const(-math.inf, elt)))
    el.emit(Assign(nan_f, ops.FALSE))
    i = Var("i", I)
    el.nonneg.add(i.uid)
    sc = el.new_scope("loop")
    body: list = []
    with el.in_block(body, sc, loop=True):
      cx = el.tmp(el.elem(coords, (i, Const(0, I))), "cx")
      cy = el.tmp(el.elem(coords, (i, Const(1, I))), "cy")
      cz = el.tmp(el.elem(coords, (i, Const(2, I))), "cz")
      any_nan = ops.lor(
        ops.prim("isnan", (cx,), Bool),
        ops.lor(ops.prim("isnan", (cy,), Bool), ops.prim("isnan", (cz,), Bool)),
      )
      el.emit(Assign(nan_f, ops.lor(nan_f, any_nan)))
      el.emit(Assign(mn_x, ops.prim("min", (mn_x, cx), elt)))
      el.emit(Assign(mn_y, ops.prim("min", (mn_y, cy), elt)))
      el.emit(Assign(mn_z, ops.prim("min", (mn_z, cz), elt)))
      el.emit(Assign(mx_x, ops.prim("max", (mx_x, cx), elt)))
      el.emit(Assign(mx_y, ops.prim("max", (mx_y, cy), elt)))
      el.emit(Assign(mx_z, ops.prim("max", (mx_z, cz), elt)))
    el.emit(ForRange(i, t0, t1, ONE, body, sc))
    for k, (mnv, mxv) in enumerate(((mn_x, mx_x), (mn_y, mx_y), (mn_z, mx_z))):
      el.emit(Store(mins.buf, el.index_expr(mins, (t, Const(k, I))), mnv))
      el.emit(Store(maxs.buf, el.index_expr(maxs, (t, Const(k, I))), mxv))
    el.emit(Store(has_nan.buf, t, nan_f))

  el.index_map((n_tiles,), build_t, ordered=False)
  return mins, maxs, has_nan, n_tiles


def _aabb_may_interact(el, qmin, qmax, qnan, qt, kmin, kmax, knan, kt, r2: Expr, elt) -> Expr:
  """Return True if 3D AABBs ``(qt, kt)`` are within ``r2`` OR either tile contains a NaN."""
  zero_f = Const(0.0, elt)
  acc = zero_f
  for k in range(3):
    kc = Const(k, I)
    d1 = ops.prim("sub", (el.elem(qmin, (qt, kc)), el.elem(kmax, (kt, kc))), elt)
    d2 = ops.prim("sub", (el.elem(kmin, (kt, kc)), el.elem(qmax, (qt, kc))), elt)
    dk = ops.prim("max", (zero_f, ops.prim("max", (d1, d2), elt)), elt)
    acc = ops.prim("add", (acc, ops.prim("mul", (dk, dk), elt)), elt)
  any_nan = ops.lor(el.elem(qnan, (qt,)), el.elem(knan, (kt,)))
  return ops.lor(any_nan, ops.prim("le", (acc, r2), Bool))


@libfunction("spatial_cutoff_sum")
def spatial_cutoff_sum(el, coords_q, coords_k, values, cutoff, weight_fn, *, tile_size=32):
  """3D AABB-pruned pairwise cutoff reduction: ``out[i] = sum_{||x_i - y_j|| <= cutoff} weight_fn(d2) * values[j]``.

  Skips provably disjoint 3D AABB tiles (preserving exact NaN semantics and zero
  ``O(N * M)`` memory).
  """
  cq = el.as_concrete(el.as_array(coords_q, "coords_q"))
  ck = el.as_concrete(el.as_array(coords_k, "coords_k"))
  vals = el.as_concrete(el.as_array(values, "values"))
  if cq.rank != 2 or ck.rank != 2 or cq.elt.kind != "f" or ck.elt.kind != "f":
    raise el.error(TypeInferenceError, "spatial_cutoff_sum() expects (M, 3) and (N, 3) float coordinate arrays")
  if el.options.bounds_check:
    el.check(cmp("eq", cq.shape[1], Const(3, I)), "ValueError", "spatial_cutoff_sum() requires coords_q shape (M, 3)")
    el.check(cmp("eq", ck.shape[1], Const(3, I)), "ValueError", "spatial_cutoff_sum() requires coords_k shape (N, 3)")
    el.check(cmp("eq", vals.shape[0], ck.shape[0]), "ValueError", "spatial_cutoff_sum() values/coords_k lengths differ")
  elt = cq.elt
  if ck.elt is not elt:
    ck = el.as_concrete(el.cast_array(ck, elt))
  if vals.elt is not elt:
    vals = el.as_concrete(el.cast_array(vals, elt))
  ts = el.static_int(tile_size, "tile_size")
  if ts <= 0:
    raise el.error(TypeInferenceError, "tile_size must be positive")
  cut = el.tmp(cast(el.as_scalar(cutoff, "cutoff"), elt), "cut")
  if el.tape is not None and not el.tape.muted and any(el.tape.is_active(el, x) for x in (cq, ck, vals, cut)):
    raise el.error(UnsupportedError, "differentiating through spatial_cutoff_sum() is not supported")
  r2 = el.tmp(
    ops.select(cmp("lt", cut, Const(0.0, elt)), Const(-1.0, elt), ops.prim("mul", (cut, cut), elt)),
    "r2",
  )

  M, N = cq.shape[0], ck.shape[0]
  qmin, qmax, qnan, _ = _build_tile_aabbs(el, cq, ts)
  kmin, kmax, knan, n_ktiles = _build_tile_aabbs(el, ck, ts)

  trailing = vals.shape[1:]
  out = el.alloc_array(elt, (M, *trailing), label="sp_sum")
  bs = Const(ts, I)

  def eval_q(idx):
    qi = idx[0]
    t_idx = idx[1:]
    qt = el.tmp(idiv(qi, bs), "qt")
    qx = el.tmp(el.elem(cq, (qi, Const(0, I))), "qx")
    qy = el.tmp(el.elem(cq, (qi, Const(1, I))), "qy")
    qz = el.tmp(el.elem(cq, (qi, Const(2, I))), "qz")
    acc = Var("acc", elt)
    el.mutables.add(acc.uid)
    el.emit(Assign(acc, Const(0.0, elt)))

    kt = Var("kt", I)
    el.nonneg.add(kt.uid)
    kt_sc = el.new_scope("loop")
    kt_body: list = []
    with el.in_block(kt_body, kt_sc, loop=True):
      may = _aabb_may_interact(el, qmin, qmax, qnan, qt, kmin, kmax, knan, kt, r2, elt)
      tblk: list = []
      with el.in_block(tblk):
        k0 = el.tmp(imul(kt, bs), "k0")
        k1 = el.tmp(imin(iadd(k0, bs), N), "k1")
        kj = Var("kj", I)
        el.nonneg.add(kj.uid)
        kj_sc = el.new_scope("loop")
        kj_body: list = []
        with el.in_block(kj_body, kj_sc, loop=True):
          dx = ops.prim("sub", (qx, el.elem(ck, (kj, Const(0, I)))), elt)
          dy = ops.prim("sub", (qy, el.elem(ck, (kj, Const(1, I)))), elt)
          dz = ops.prim("sub", (qz, el.elem(ck, (kj, Const(2, I)))), elt)
          d2 = el.tmp(
            ops.prim(
              "add",
              (
                ops.prim("add", (ops.prim("mul", (dx, dx), elt), ops.prim("mul", (dy, dy), elt)), elt),
                ops.prim("mul", (dz, dz), elt),
              ),
              elt,
            ),
            "d2",
          )
          in_r = ops.prim("le", (d2, r2), Bool)
          w_blk: list = []
          with el.in_block(w_blk):
            w = cast(el.as_scalar(el.call(weight_fn, [d2]), "weight_fn"), elt)
            el.emit(Assign(acc, ops.prim("add", (acc, ops.prim("mul", (w, el.elem(vals, (kj, *t_idx))), elt)), elt)))
          el.emit(If(in_r, w_blk, []))
        el.emit(ForRange(kj, k0, k1, ONE, kj_body, kj_sc))
      el.emit(If(may, tblk, []))
    el.emit(ForRange(kt, ZERO, n_ktiles, ONE, kt_body, kt_sc))
    el.emit(Store(out.buf, el.index_expr(out, idx), acc))

  el.index_map((M, *trailing), eval_q, ordered=False)
  return out


# ---------------------------------------------------------------------------
# 3. CSR BlockMask block-sparse FlashAttention (Forward + Law 2 Backward)
# ---------------------------------------------------------------------------


def _record_ad_node(tape, el, kind: str, inputs: list, out: ArrayVal, info) -> None:
  from ..transforms.autodiff import Node

  node = Node(kind, tuple(inputs), info)
  tape.record(el, node)
  tape.register(el, node, out)


def _normalize_qkv_3d_batch(el, x: ArrayVal, what: str) -> tuple[ArrayVal, tuple[Expr, ...]]:
  """Reshape ``(*batch, S, D)`` into ``(B_flat, S, D)``, returning ``(x3, batch_shape)``."""
  if x.rank < 2 or x.elt.kind != "f":
    raise el.error(TypeInferenceError, f"{what} must be >=2D floating-point array, got {describe(x)}")
  batch = x.shape[:-2]
  b_flat = ops.iprod(batch) if batch else ONE
  x3 = el.as_concrete(el.reshape(x, (b_flat, x.shape[-2], x.shape[-1]), checked=False))
  return x3, batch


@libfunction("block_sparse_attention")
def block_sparse_attention(el, q, k, v, block_mask, *, scale=None, causal=False, mask=None):
  """Law 5 CSR ``BlockMask`` sparse FlashAttention (forward + Law 2 ``D_i``-recomputed backward).

  Accepts ``q (*batch, M, K)``, ``k (*batch, N, K)``, ``v (*batch, N, D)`` and a
  :class:`~parakeet.BlockMask`.  Iterates only active CSR key blocks per query
  block, handles empty rows cleanly (output 0), supports optional fine-grained
  ``causal=True`` or elementwise boolean ``mask (M, N)`` inside boundary blocks,
  and differentiates via Law 2 ``D_i``-recomputed backward using CSC ``(col_ptr, row_idx)``.
  """
  if not isinstance(block_mask, BlockMask):
    raise el.error(TypeInferenceError, f"block_sparse_attention() expects a BlockMask, got {describe(block_mask)}")
  q_arr = el.as_array(q, "q")
  k_arr = el.as_array(k, "k")
  v_arr = el.as_array(v, "v")
  q3, batch = _normalize_qkv_3d_batch(el, q_arr, "q")
  k3, _ = _normalize_qkv_3d_batch(el, k_arr, "k")
  v3, _ = _normalize_qkv_3d_batch(el, v_arr, "v")
  elt = q3.elt
  if k3.elt is not elt:
    k3 = el.as_concrete(el.cast_array(k3, elt))
  if v3.elt is not elt:
    v3 = el.as_concrete(el.cast_array(v3, elt))

  B_flat, M, K_dim = q3.shape
  _, N, _ = k3.shape
  _, _, D_dim = v3.shape
  bq = el.static_int(block_mask.block_q, "block_q")
  bk = el.static_int(block_mask.block_k, "block_k")
  is_causal = el.static_bool(causal, "causal") if not isinstance(causal, bool) else causal
  mask_arr = None if mask is None else el.as_concrete(el.as_array(mask, "mask"))
  if el.options.bounds_check:
    el.check(cmp("eq", k3.shape[0], B_flat), "ValueError", "block_sparse_attention() q and k batch sizes differ")
    el.check(cmp("eq", v3.shape[0], B_flat), "ValueError", "block_sparse_attention() q and v batch sizes differ")
    el.check(cmp("eq", k3.shape[2], K_dim), "ValueError", "block_sparse_attention() q and k feature dims differ")
    el.check(cmp("eq", v3.shape[1], N), "ValueError", "block_sparse_attention() k and v sequence lengths differ")
    if mask_arr is not None:
      if mask_arr.rank != 2:
        raise el.error(TypeInferenceError, "block_sparse_attention() mask must be 2D (M, N)")
      el.check(cmp("eq", mask_arr.shape[0], M), "ValueError", "block_sparse_attention() mask rows != M")
      el.check(cmp("eq", mask_arr.shape[1], N), "ValueError", "block_sparse_attention() mask cols != N")
  if scale is None:
    sc = el.tmp(
      ops.select(
        cmp("gt", K_dim, ZERO),
        ops.prim("div", (Const(1.0, elt), ops.prim("sqrt", (cast(K_dim, elt),), elt)), elt),
        Const(0.0, elt),
      ),
      "scale",
    )
  else:
    sc = el.tmp(cast(el.as_scalar(scale, "scale"), elt), "scale")

  row_ptr = el.as_concrete(block_mask.row_ptr)
  col_idx = el.as_concrete(block_mask.col_idx)
  col_ptr = el.as_concrete(block_mask.col_ptr)
  row_idx = el.as_concrete(block_mask.row_idx)

  tape = el.tape
  if tape is not None and not tape.muted and tape.is_active(el, sc):
    raise el.error(UnsupportedError, "differentiating block_sparse_attention() with respect to scale is not supported")
  record_ad = (
    tape is not None and not tape.muted and (tape.is_active(el, q3) or tape.is_active(el, k3) or tape.is_active(el, v3))
  )

  saved_tape = el.tape
  if record_ad:
    el.tape = None
  try:
    out = el.alloc_array(elt, (B_flat, M, D_dim), zero=True, label="bsa_out")
    lse = el.alloc_array(elt, (B_flat, M), zero=True, label="bsa_lse") if record_ad else None
    bq_c = Const(bq, I)
    bk_c = Const(bk, I)

    def fwd_row(idx):
      b, qi = idx
      qb = el.tmp(idiv(qi, bq_c), "qb")
      ps = el.tmp(cast(el.elem(row_ptr, (qb,)), I), "ps")
      pe = el.tmp(cast(el.elem(row_ptr, (iadd(qb, ONE),)), I), "pe")

      m = Var("m", elt)
      l_acc = Var("l", elt)
      el.mutables.update((m.uid, l_acc.uid))
      el.emit(Assign(m, Const(-math.inf, elt)))
      el.emit(Assign(l_acc, Const(0.0, elt)))

      p = Var("p", I)
      el.nonneg.add(p.uid)
      psc = el.new_scope("loop")
      pbody: list = []
      with el.in_block(pbody, psc, loop=True):
        kb = el.tmp(cast(el.elem(col_idx, (p,)), I), "kb")
        k0 = el.tmp(imul(kb, bk_c), "k0")
        k1 = el.tmp(imin(iadd(k0, bk_c), N), "k1")
        if is_causal:
          k1 = el.tmp(imin(k1, iadd(qi, ONE)), "k1c")
        kj = Var("kj", I)
        el.nonneg.add(kj.uid)
        ksc = el.new_scope("loop")
        kbody: list = []
        with el.in_block(kbody, ksc, loop=True):
          inner_blk: list = []
          with el.in_block(inner_blk):
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
                  dot,
                  ops.prim("add", (dot, ops.prim("mul", (el.elem(q3, (b, qi, c)), el.elem(k3, (b, kj, c))), elt)), elt),
                )
              )
            el.emit(ForRange(c, ZERO, K_dim, ONE, cb, csc))
            score = el.tmp(ops.prim("mul", (dot, sc), elt), "score")
            m_new = el.tmp(ops.prim("max", (m, score), elt), "mn")
            alpha = el.tmp(ops.prim("exp", (ops.prim("sub", (m, m_new), elt),), elt), "alpha")
            beta = el.tmp(ops.prim("exp", (ops.prim("sub", (score, m_new), elt),), elt), "beta")
            el.emit(Assign(l_acc, ops.prim("add", (ops.prim("mul", (l_acc, alpha), elt), beta), elt)))
            el.emit(Assign(m, m_new))
            d = Var("d", I)
            el.nonneg.add(d.uid)
            dsc = el.new_scope("loop")
            db: list = []
            with el.in_block(db, dsc, loop=True):
              opos = el.index_expr(out, (b, qi, d))
              el.emit(
                Store(
                  out.buf,
                  opos,
                  ops.prim(
                    "add",
                    (
                      ops.prim("mul", (Load(out.buf, opos), alpha), elt),
                      ops.prim("mul", (beta, el.elem(v3, (b, kj, d))), elt),
                    ),
                    elt,
                  ),
                )
              )
            el.emit(ForRange(d, ZERO, D_dim, ONE, db, dsc))
          if mask_arr is not None:
            el.emit(If(cast(el.elem(mask_arr, (qi, kj)), Bool), inner_blk, []))
          else:
            el.block.extend(inner_blk)
        el.emit(ForRange(kj, k0, k1, ONE, kbody, ksc))
      el.emit(ForRange(p, ps, pe, ONE, pbody, psc))

      has_any = cmp("gt", l_acc, Const(0.0, elt))
      inv_l = el.tmp(ops.select(has_any, ops.prim("div", (Const(1.0, elt), l_acc), elt), Const(0.0, elt)), "inv_l")
      d2 = Var("d2", I)
      el.nonneg.add(d2.uid)
      d2sc = el.new_scope("loop")
      d2b: list = []
      with el.in_block(d2b, d2sc, loop=True):
        opos = el.index_expr(out, (b, qi, d2))
        el.emit(Store(out.buf, opos, ops.prim("mul", (Load(out.buf, opos), inv_l), elt)))
      el.emit(ForRange(d2, ZERO, D_dim, ONE, d2b, d2sc))
      if lse is not None:
        lse_v = ops.select(has_any, ops.prim("add", (m, ops.prim("log", (l_acc,), elt)), elt), Const(-math.inf, elt))
        el.emit(Store(lse.buf, el.index_expr(lse, (b, qi)), lse_v))

    el.index_map((B_flat, M), fwd_row, ordered=False)
  finally:
    el.tape = saved_tape

  if record_ad:
    _record_ad_node(
      tape,
      el,
      "block_sparse_attention",
      [q3, k3, v3],
      out,
      (row_ptr, col_idx, col_ptr, row_idx, bq, bk, out, lse, sc, is_causal, mask_arr),
    )

  return el.reshape(out, (*batch, M, D_dim), checked=False)


def _block_sparse_attention_vjp(tape, el, node, g: ArrayVal) -> None:
  """Law 2 ``D_i``-recomputed CSR/CSC block-sparse FlashAttention backward (zero atomics)."""
  q3, k3, v3 = node.inputs
  row_ptr, col_idx, col_ptr, row_idx, bq, bk, fwd_out, lse, sc, is_causal, mask_arr = node.info
  tape.stale(el, node, q3)
  tape.stale(el, node, k3)
  tape.stale(el, node, v3)
  g = el.as_concrete(el.cast_array(g, q3.elt))
  elt = q3.elt
  B_flat, M, K_dim = q3.shape
  _, N, D_dim = v3.shape
  bq_c = Const(bq, I)
  bk_c = Const(bk, I)

  need_q = tape.is_active(el, q3)
  need_k = tape.is_active(el, k3)
  need_v = tape.is_active(el, v3)

  # D[b, qi] = sum_d dO[b, qi, d] * O[b, qi, d]
  d_vec = el.alloc_array(elt, (B_flat, M), label="bsa_D")

  def compute_d(idx):
    b, qi = idx
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
          acc, ops.prim("add", (acc, ops.prim("mul", (el.elem(g, (b, qi, d)), el.elem(fwd_out, (b, qi, d))), elt)), elt)
        )
      )
    el.emit(ForRange(d, ZERO, D_dim, ONE, db, dsc))
    el.emit(Store(d_vec.buf, el.index_expr(d_vec, (b, qi)), acc))

  el.index_map((B_flat, M), compute_d, ordered=False)

  # Atomic-free dQ iterating CSR row_ptr[qb] .. row_ptr[qb + 1]
  if need_q:
    dq = el.alloc_array(elt, q3.shape, zero=True, label="bsa_dQ")

    def compute_dq(idx):
      b, qi = idx
      qb = el.tmp(idiv(qi, bq_c), "qb")
      ps = el.tmp(cast(el.elem(row_ptr, (qb,)), I), "ps")
      pe = el.tmp(cast(el.elem(row_ptr, (iadd(qb, ONE),)), I), "pe")
      lse_i = el.tmp(el.elem(lse, (b, qi)), "lse_i")
      d_i = el.tmp(el.elem(d_vec, (b, qi)), "d_i")
      p = Var("p", I)
      el.nonneg.add(p.uid)
      psc = el.new_scope("loop")
      pbody: list = []
      with el.in_block(pbody, psc, loop=True):
        kb = el.tmp(cast(el.elem(col_idx, (p,)), I), "kb")
        k0 = el.tmp(imul(kb, bk_c), "k0")
        k1 = el.tmp(imin(iadd(k0, bk_c), N), "k1")
        if is_causal:
          k1 = el.tmp(imin(k1, iadd(qi, ONE)), "k1c")
        kj = Var("kj", I)
        el.nonneg.add(kj.uid)
        ksc = el.new_scope("loop")
        kbody: list = []
        with el.in_block(kbody, ksc, loop=True):
          inner: list = []
          with el.in_block(inner):
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
                  dot,
                  ops.prim("add", (dot, ops.prim("mul", (el.elem(q3, (b, qi, c)), el.elem(k3, (b, kj, c))), elt)), elt),
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
                  dp,
                  ops.prim("add", (dp, ops.prim("mul", (el.elem(g, (b, qi, d)), el.elem(v3, (b, kj, d))), elt)), elt),
                )
              )
            el.emit(ForRange(d, ZERO, D_dim, ONE, db, dsc))
            ds = el.tmp(ops.prim("mul", (ops.prim("mul", (p_ij, ops.prim("sub", (dp, d_i), elt)), elt), sc), elt), "ds")
            c2 = Var("c2", I)
            el.nonneg.add(c2.uid)
            c2sc = el.new_scope("loop")
            c2b: list = []
            with el.in_block(c2b, c2sc, loop=True):
              qpos = el.index_expr(dq, (b, qi, c2))
              el.emit(
                Store(
                  dq.buf,
                  qpos,
                  ops.prim("add", (Load(dq.buf, qpos), ops.prim("mul", (ds, el.elem(k3, (b, kj, c2))), elt)), elt),
                )
              )
            el.emit(ForRange(c2, ZERO, K_dim, ONE, c2b, c2sc))
          if mask_arr is not None:
            el.emit(If(cast(el.elem(mask_arr, (qi, kj)), Bool), inner, []))
          else:
            el.block.extend(inner)
        el.emit(ForRange(kj, k0, k1, ONE, kbody, ksc))
      el.emit(ForRange(p, ps, pe, ONE, pbody, psc))

    el.index_map((B_flat, M), compute_dq, ordered=False)
    tape.add(el, q3, dq)

  # Atomic-free dK & dV iterating CSC col_ptr[kb] .. col_ptr[kb + 1]
  if need_k or need_v:
    dk = el.alloc_array(elt, k3.shape, zero=True, label="bsa_dK") if need_k else None
    dv = el.alloc_array(elt, v3.shape, zero=True, label="bsa_dV") if need_v else None

    def compute_dkv(idx):
      b, kj = idx
      kb = el.tmp(idiv(kj, bk_c), "kb")
      ps = el.tmp(cast(el.elem(col_ptr, (kb,)), I), "ps")
      pe = el.tmp(cast(el.elem(col_ptr, (iadd(kb, ONE),)), I), "pe")
      p = Var("p", I)
      el.nonneg.add(p.uid)
      psc = el.new_scope("loop")
      pbody: list = []
      with el.in_block(pbody, psc, loop=True):
        qb = el.tmp(cast(el.elem(row_idx, (p,)), I), "qb")
        q0 = el.tmp(imul(qb, bq_c), "q0")
        q1 = el.tmp(imin(iadd(q0, bq_c), M), "q1")
        if is_causal:
          q0 = el.tmp(imax(q0, kj), "q0c")
        qi = Var("qi", I)
        el.nonneg.add(qi.uid)
        qsc = el.new_scope("loop")
        qbody: list = []
        with el.in_block(qbody, qsc, loop=True):
          inner: list = []
          with el.in_block(inner):
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
                  dot,
                  ops.prim("add", (dot, ops.prim("mul", (el.elem(q3, (b, qi, c)), el.elem(k3, (b, kj, c))), elt)), elt),
                )
              )
            el.emit(ForRange(c, ZERO, K_dim, ONE, cb, csc))
            lse_i = el.elem(lse, (b, qi))
            p_ij = el.tmp(ops.prim("exp", (ops.prim("sub", (ops.prim("mul", (dot, sc), elt), lse_i), elt),), elt), "p")
            if dv is not None:
              d = Var("d", I)
              el.nonneg.add(d.uid)
              dsc = el.new_scope("loop")
              db: list = []
              with el.in_block(db, dsc, loop=True):
                vpos = el.index_expr(dv, (b, kj, d))
                el.emit(
                  Store(
                    dv.buf,
                    vpos,
                    ops.prim("add", (Load(dv.buf, vpos), ops.prim("mul", (p_ij, el.elem(g, (b, qi, d))), elt)), elt),
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
                    ops.prim(
                      "add", (dp, ops.prim("mul", (el.elem(g, (b, qi, d2)), el.elem(v3, (b, kj, d2))), elt)), elt
                    ),
                  )
                )
              el.emit(ForRange(d2, ZERO, D_dim, ONE, d2b, d2sc))
              d_i = el.elem(d_vec, (b, qi))
              ds = el.tmp(
                ops.prim("mul", (ops.prim("mul", (p_ij, ops.prim("sub", (dp, d_i), elt)), elt), sc), elt), "ds"
              )
              c2 = Var("c2", I)
              el.nonneg.add(c2.uid)
              c2sc = el.new_scope("loop")
              c2b: list = []
              with el.in_block(c2b, c2sc, loop=True):
                kpos = el.index_expr(dk, (b, kj, c2))
                el.emit(
                  Store(
                    dk.buf,
                    kpos,
                    ops.prim("add", (Load(dk.buf, kpos), ops.prim("mul", (ds, el.elem(q3, (b, qi, c2))), elt)), elt),
                  )
                )
              el.emit(ForRange(c2, ZERO, K_dim, ONE, c2b, c2sc))
          if mask_arr is not None:
            el.emit(If(cast(el.elem(mask_arr, (qi, kj)), Bool), inner, []))
          else:
            el.block.extend(inner)
        el.emit(ForRange(qi, q0, q1, ONE, qbody, qsc))
      el.emit(ForRange(p, ps, pe, ONE, pbody, psc))

    el.index_map((B_flat, N), compute_dkv, ordered=False)
    if dk is not None:
      tape.add(el, k3, dk)
    if dv is not None:
      tape.add(el, v3, dv)
