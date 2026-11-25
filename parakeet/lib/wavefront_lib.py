"""2D / banded anti-diagonal wavefront semiring DP and checkpointed differentiable alignment."""

from __future__ import annotations

import math

from ..errors import TypeInferenceError, UnsupportedError
from ..ndtypes import Float64, Int64
from ..syntax import ops
from ..syntax.ir import Assign, Const, Expr, ForRange, If, IndexWavefront, Load, Scope, Store, Var
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imul, isub
from ..syntax.records import record_type
from ..syntax.values import ArrayVal, describe
from .adverbs import libfunction


def _flatten_state(el, v, what: str) -> tuple[list[Expr], object]:
  if isinstance(v, Expr):
    return [v], lambda xs: xs[0]
  if isinstance(v, ArrayVal) and v.rank == 0:
    return [el.as_scalar(v, what)], lambda xs: xs[0]
  if rec := record_type(v):
    leaves, rb = _flatten_state(el, rec.values(v), what)
    return leaves, lambda xs, _r=rec, _rb=rb: _r.build(_rb(xs))
  if isinstance(v, tuple):
    parts = [_flatten_state(el, x, what) for x in v]
    flat: list[Expr] = []
    cnts: list[int] = []
    for ls, _ in parts:
      flat.extend(ls)
      cnts.append(len(ls))

    def rebuild(xs, _parts=parts, _cnts=cnts):
      out = []
      pos = 0
      for (_, rb), c in zip(_parts, _cnts):
        out.append(rb(xs[pos : pos + c]))
        pos += c
      return tuple(out)

    return flat, rebuild
  raise el.error(TypeInferenceError, f"{what} must be a scalar or tuple/record of scalars, got {describe(v)}")


@libfunction("wavefront")
def wavefront(el, step_fn, shape, init_top, init_left, *, init_corner=None, band=None, track_max=False):
  """2D / banded anti-diagonal wavefront dynamic programming (O(M) memory, zero M x N table).

  ``step_fn(top, left, diag, i, j) -> next_state`` computes cell ``(i, j)`` from its
  three causal predecessors.  State may be a scalar or a ``NamedTuple`` / frozen
  ``@dataclass`` / tuple of scalars (e.g. 3-state Gotoh affine-gap alignment).
  When ``track_max=True``, returns ``(final_state, max_val, best_i, best_j)``.
  """
  sh = el.shape_tuple(shape, "wavefront shape")
  if len(sh) != 2:
    raise el.error(TypeInferenceError, f"wavefront() shape must have 2 dimensions, got {len(sh)}")
  m, n = el.tmp(sh[0], "m"), el.tmp(sh[1], "n")
  top_leaves, rebuild = _flatten_state(el, init_top, "init_top")
  left_leaves, _ = _flatten_state(el, init_left, "init_left")
  corner_leaves = top_leaves if init_corner is None else _flatten_state(el, init_corner, "init_corner")[0]
  if not (len(top_leaves) == len(left_leaves) == len(corner_leaves)):
    raise el.error(TypeInferenceError, "wavefront() init_top, init_left, and init_corner must have matching shapes")

  from ..ndtypes import join

  field_types = [join(join(a.type, b.type), c.type).strong for a, b, c in zip(top_leaves, left_leaves, corner_leaves)]
  k_fields = len(field_types)
  band_e = None if band is None else el.as_index(band, "band")
  do_max = el.static_bool(track_max, "track_max") if not isinstance(track_max, bool) else track_max

  i_var = Var("wf_i", I)
  j_var = Var("wf_j", I)
  el.nonneg.update((i_var.uid, j_var.uid))
  el.upper[i_var.uid] = m
  el.upper[j_var.uid] = n

  top_vars = tuple(Var(f"top_{f}", field_types[f]) for f in range(k_fields))
  left_vars = tuple(Var(f"left_{f}", field_types[f]) for f in range(k_fields))
  diag_vars = tuple(Var(f"diag_{f}", field_types[f]) for f in range(k_fields))

  wf_scope = el.new_scope("wavefront")
  body: list = []
  with el.in_block(body, wf_scope, loop=True):
    nxt = el.call(step_fn, [rebuild(top_vars), rebuild(left_vars), rebuild(diag_vars), i_var, j_var])
    nxt_leaves, _ = _flatten_state(el, nxt, "wavefront step_fn")
    if len(nxt_leaves) != k_fields:
      raise el.error(TypeInferenceError, f"wavefront step_fn returned {len(nxt_leaves)} fields, expected {k_fields}")
    nxt_cast = tuple(cast(v, field_types[f]) for f, v in enumerate(nxt_leaves))

  res_vars = tuple(Var(f"wf_res{f}", field_types[f]) for f in range(k_fields))
  for rv in res_vars:
    el.mutables.add(rv.uid)

  max_var = best_i = best_j = None
  if do_max:
    max_var = Var("wf_max", field_types[0])
    best_i = Var("wf_bi", Int64)
    best_j = Var("wf_bj", Int64)
    el.mutables.update((max_var.uid, best_i.uid, best_j.uid))

  el.emit(
    IndexWavefront(
      m,
      n,
      i_var,
      j_var,
      top_vars,
      left_vars,
      diag_vars,
      tuple(cast(x, field_types[f]) for f, x in enumerate(top_leaves)),
      tuple(cast(x, field_types[f]) for f, x in enumerate(left_leaves)),
      tuple(cast(x, field_types[f]) for f, x in enumerate(corner_leaves)),
      body,
      nxt_cast,
      res_vars,
      wf_scope,
      band=band_e,
      track_max=do_max,
      max_val=max_var,
      best_i=best_i,
      best_j=best_j,
      loc=el.loc(),
    )
  )
  final_state = rebuild(res_vars)
  if do_max:
    return (final_state, max_var, best_i, best_j)
  return final_state


@libfunction("batched_wavefront")
def batched_wavefront(
  el, step_fn, m_lens, n_lens, init_top, init_left, *, init_corner=None, band=None, track_max=False
):
  """GPU- and multicore-friendly batched 2D anti-diagonal DP over ``B`` variable-length pairs.

  Allocates a single contiguous ``(B, 3, max_m)`` ring buffer before the parallel
  batch loop so all ``B`` alignments run concurrently on CPU, CUDA, and Apple Metal
  GPUs with zero per-thread allocation.
  ``step_fn(b, top, left, diag, i, j) -> next_state``.
  """
  ms = el.as_concrete(el.as_array(m_lens, "m_lens"))
  ns = el.as_concrete(el.as_array(n_lens, "n_lens"))
  if ms.rank != 1 or ns.rank != 1:
    raise el.error(TypeInferenceError, "batched_wavefront() expects 1D length arrays m_lens and n_lens")
  B = ms.shape[0]
  if el.options.bounds_check:
    el.check(cmp("eq", B, ns.shape[0]), "ValueError", "m_lens and n_lens must have the same batch length")

  top_leaves, rebuild = _flatten_state(el, init_top, "init_top")
  left_leaves, _ = _flatten_state(el, init_left, "init_left")
  corner_leaves = top_leaves if init_corner is None else _flatten_state(el, init_corner, "init_corner")[0]
  from ..ndtypes import join

  field_types = [join(join(a.type, b.type), c.type).strong for a, b, c in zip(top_leaves, left_leaves, corner_leaves)]
  k_fields = len(field_types)
  band_e = None if band is None else el.as_index(band, "band")
  do_max = el.static_bool(track_max, "track_max") if not isinstance(track_max, bool) else track_max

  # Compute max_m = max(1, max(m_lens)) once outside the parallel loop (safe on empty batch B == 0)
  from .numpy_lib import amax

  max_m = Var("max_m", I)
  el.mutables.add(max_m.uid)
  el.emit(Assign(max_m, ONE))
  bm_blk: list = []
  with el.in_block(bm_blk):
    el.emit(Assign(max_m, imax(ONE, cast(el.as_scalar(amax(el, ms, None, False)), I))))
  el.emit(If(cmp("gt", B, ZERO), bm_blk, []))
  stride_b = el.tmp(imul(Const(3, I), max_m), "wf_sb")
  rings = [el.alloc_array(field_types[f], (imul(B, stride_b),), label=f"bwf_r{f}") for f in range(k_fields)]
  out_fields = [el.alloc_array(field_types[f], (B,), label=f"bwf_out{f}") for f in range(k_fields)]
  out_max = el.alloc_array(field_types[0], (B,), label="bwf_max") if do_max else None
  out_bi = el.alloc_array(Int64, (B,), label="bwf_bi") if do_max else None
  out_bj = el.alloc_array(Int64, (B,), label="bwf_bj") if do_max else None

  def run_one(idx):
    b = idx[0]
    m = el.tmp(cast(el.elem(ms, (b,)), I), "m")
    n = el.tmp(cast(el.elem(ns, (b,)), I), "n")
    b_off = el.tmp(imul(b, stride_b), "boff")
    for f in range(k_fields):
      el.emit(Store(out_fields[f].buf, b, cast(corner_leaves[f], field_types[f])))
    if do_max:
      el.emit(Store(out_max.buf, b, cast(corner_leaves[0], field_types[0])))
      el.emit(Store(out_bi.buf, b, Const(-1, Int64)))
      el.emit(Store(out_bj.buf, b, Const(-1, Int64)))

    non_empty: list = []
    with el.in_block(non_empty):
      mval = Var("bmv", field_types[0]) if do_max else None
      bi = Var("bbi", Int64) if do_max else None
      bj = Var("bbj", Int64) if do_max else None
      if do_max:
        el.mutables.update((mval.uid, bi.uid, bj.uid))
        el.emit(Assign(mval, cast(corner_leaves[0], field_types[0])))
        el.emit(Assign(bi, Const(-1, Int64)))
        el.emit(Assign(bj, Const(-1, Int64)))
      num_d = el.tmp(isub(iadd(m, n), ONE), "nd")
      d = Var("d", I)
      el.nonneg.add(d.uid)
      dsc = el.new_scope("loop")
      dbody: list = []
      with el.in_block(dbody, dsc, loop=True):
        cs = el.tmp(iadd(b_off, imul(ops.imod(d, Const(3, I)), max_m)), "cs")
        p1 = el.tmp(iadd(b_off, imul(ops.imod(iadd(d, Const(2, I)), Const(3, I)), max_m)), "p1")
        p2 = el.tmp(iadd(b_off, imul(ops.imod(iadd(d, ONE), Const(3, I)), max_m)), "p2")
        ilo = imax(ZERO, isub(d, isub(n, ONE)))
        ihi = imin(isub(m, ONE), d)
        if band_e is not None:
          ilo = imax(ilo, idiv(iadd(isub(d, band_e), ONE), Const(2, I)))
          ihi = imin(ihi, idiv(iadd(d, band_e), Const(2, I)))
        ilo_v = el.tmp(ilo, "ilo")
        ihi_v = el.tmp(iadd(ihi, ONE), "ihi")
        iv = Var("i", I)
        el.nonneg.add(iv.uid)
        isc = el.new_scope("loop")
        ibody: list = []
        with el.in_block(ibody, isc, loop=True):
          jv = el.tmp(isub(d, iv), "j")
          ht = cmp("gt", iv, ZERO)
          hl = cmp("gt", jv, ZERO)
          if band_e is not None:
            ht = ops.land(ht, cmp("le", isub(jv, isub(iv, ONE)), band_e))
            hl = ops.land(hl, cmp("le", isub(iv, isub(jv, ONE)), band_e))
          hd = ops.land(cmp("gt", iv, ZERO), cmp("gt", jv, ZERO))
          sim1 = el.tmp(imax(isub(iv, ONE), ZERO), "im1")
          tvs, lvs, dvs = [], [], []
          for f in range(k_fields):
            ft = field_types[f]
            rb = rings[f].buf
            df = ops.select(
              cmp("gt", jv, ZERO),
              cast(top_leaves[f], ft),
              ops.select(cmp("gt", iv, ZERO), cast(left_leaves[f], ft), cast(corner_leaves[f], ft)),
            )
            tvs.append(el.tmp(ops.select(ht, Load(rb, iadd(p1, sim1)), cast(top_leaves[f], ft)), f"t{f}"))
            lvs.append(el.tmp(ops.select(hl, Load(rb, iadd(p1, iv)), cast(left_leaves[f], ft)), f"l{f}"))
            dvs.append(el.tmp(ops.select(hd, Load(rb, iadd(p2, sim1)), df), f"dg{f}"))
          nxt = el.call(step_fn, [b, rebuild(tvs), rebuild(lvs), rebuild(dvs), iv, jv])
          nleaves, _ = _flatten_state(el, nxt, "batched_wavefront step_fn")
          for f in range(k_fields):
            nv = el.tmp(cast(nleaves[f], field_types[f]), f"nv{f}")
            el.emit(Store(rings[f].buf, iadd(cs, iv), nv))
            if f == 0 and do_max:
              el.emit(
                If(
                  cmp("gt", nv, mval), [Assign(mval, nv), Assign(bi, cast(iv, Int64)), Assign(bj, cast(jv, Int64))], []
                )
              )
        el.emit(ForRange(iv, ilo_v, ihi_v, ONE, ibody, isc))
      el.emit(ForRange(d, ZERO, num_d, ONE, dbody, dsc))
      last_base = el.tmp(iadd(b_off, imul(ops.imod(isub(num_d, ONE), Const(3, I)), max_m)), "lb")
      last_pos = el.tmp(iadd(last_base, isub(m, ONE)), "lp")
      store_last = [Store(out_fields[f].buf, b, Load(rings[f].buf, last_pos)) for f in range(k_fields)]
      if band_e is not None:
        el.emit(If(cmp("le", imax(isub(m, n), isub(n, m)), band_e), store_last, []))
      else:
        for st in store_last:
          el.emit(st)
      if do_max:
        el.emit(Store(out_max.buf, b, mval))
        el.emit(Store(out_bi.buf, b, bi))
        el.emit(Store(out_bj.buf, b, bj))

    el.emit(If(ops.land(cmp("gt", m, ZERO), cmp("gt", n, ZERO)), non_empty, []))

  el.index_map((B,), run_one, ordered=False)
  final_out = rebuild(out_fields)
  if do_max:
    return (final_out, out_max, out_bi, out_bj)
  return final_out


# ---------------------------------------------------------------------------
# Differentiable Soft-Smith-Waterman / Soft-Needleman-Wunsch (Law 2 checkpointed)
# ---------------------------------------------------------------------------


def _lse2(a: Expr, b: Expr, inv_tau: Expr, tau: Expr, elt) -> Expr:
  m = ops.prim("max", (a, b), elt)
  da = ops.prim("mul", (ops.prim("sub", (a, m), elt), inv_tau), elt)
  db = ops.prim("mul", (ops.prim("sub", (b, m), elt), inv_tau), elt)
  s = ops.prim("add", (ops.prim("exp", (da,), elt), ops.prim("exp", (db,), elt)), elt)
  return ops.select(
    ops.prim("isfinite", (m,), ops.Bool),
    ops.prim("add", (m, ops.prim("mul", (tau, ops.prim("log", (s,), elt)), elt)), elt),
    m,
  )


def _lse3(a: Expr, b: Expr, c: Expr, inv_tau: Expr, tau: Expr, elt) -> Expr:
  m = ops.prim("max", (ops.prim("max", (a, b), elt), c), elt)
  ea = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (a, m), elt), inv_tau), elt),), elt)
  eb = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (b, m), elt), inv_tau), elt),), elt)
  ec = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (c, m), elt), inv_tau), elt),), elt)
  s = ops.prim("add", (ops.prim("add", (ea, eb), elt), ec), elt)
  return ops.select(
    ops.prim("isfinite", (m,), ops.Bool),
    ops.prim("add", (m, ops.prim("mul", (tau, ops.prim("log", (s,), elt)), elt)), elt),
    m,
  )


def _lse4(a: Expr, b: Expr, c: Expr, d: Expr, inv_tau: Expr, tau: Expr, elt) -> Expr:
  m = ops.prim("max", (ops.prim("max", (a, b), elt), ops.prim("max", (c, d), elt)), elt)
  ea = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (a, m), elt), inv_tau), elt),), elt)
  eb = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (b, m), elt), inv_tau), elt),), elt)
  ec = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (c, m), elt), inv_tau), elt),), elt)
  ed = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (d, m), elt), inv_tau), elt),), elt)
  s = ops.prim("add", (ops.prim("add", (ea, eb), elt), ops.prim("add", (ec, ed), elt)), elt)
  return ops.select(
    ops.prim("isfinite", (m,), ops.Bool),
    ops.prim("add", (m, ops.prim("mul", (tau, ops.prim("log", (s,), elt)), elt)), elt),
    m,
  )


def _emit_soft_row_step(
  el,
  scores: ArrayVal,
  gap: Expr,
  inv_tau: Expr,
  tau: Expr,
  elt,
  is_local: bool,
  band: Expr | None,
  i: Expr,
  N: Expr,
  prev_buf: Var,
  prev_off: Expr,
  cur_buf: Var,
  cur_off: Expr,
  tot_var: Var | None,
) -> None:
  """Compute row ``i`` of soft-SW / soft-NW from ``prev_buf`` into ``cur_buf``."""
  jlo = ZERO if band is None else el.tmp(imax(ZERO, isub(i, band)), "jlo")
  jhi = N if band is None else el.tmp(imin(N, iadd(iadd(i, band), ONE)), "jhi")
  zero_f = Const(0.0, elt)
  neginf = Const(-math.inf, elt)
  j = Var("j", I)
  el.nonneg.add(j.uid)
  jsc = el.new_scope("loop")
  jbody: list = []
  with el.in_block(jbody, jsc, loop=True):
    jm1 = el.tmp(imax(isub(j, ONE), ZERO), "jm1")
    top_in_band = ops.TRUE if band is None else cmp("le", isub(j, isub(i, ONE)), band)
    left_in_band = ops.TRUE if band is None else cmp("le", isub(i, isub(j, ONE)), band)

    top_bnd = zero_f if is_local else ops.prim("mul", (cast(iadd(j, ONE), elt), gap), elt)
    left_bnd = zero_f if is_local else ops.prim("mul", (cast(iadd(i, ONE), elt), gap), elt)
    diag_bnd = (
      zero_f
      if is_local
      else ops.select(
        cmp("gt", j, ZERO),
        ops.prim("mul", (cast(j, elt), gap), elt),
        ops.select(cmp("gt", i, ZERO), ops.prim("mul", (cast(i, elt), gap), elt), zero_f),
      )
    )
    oob_val = zero_f if is_local else neginf

    top_v = el.tmp(
      ops.select(cmp("gt", i, ZERO), ops.select(top_in_band, Load(prev_buf, iadd(prev_off, j)), oob_val), top_bnd),
      "topv",
    )
    left_v = el.tmp(
      ops.select(cmp("gt", j, ZERO), ops.select(left_in_band, Load(cur_buf, iadd(cur_off, jm1)), oob_val), left_bnd),
      "leftv",
    )
    diag_v = el.tmp(
      ops.select(ops.land(cmp("gt", i, ZERO), cmp("gt", j, ZERO)), Load(prev_buf, iadd(prev_off, jm1)), diag_bnd),
      "diagv",
    )

    c_diag = el.tmp(ops.prim("add", (diag_v, el.elem(scores, (i, j))), elt), "cd")
    c_top = el.tmp(ops.prim("add", (top_v, gap), elt), "ct")
    c_left = el.tmp(ops.prim("add", (left_v, gap), elt), "cl")
    if is_local:
      h_ij = el.tmp(_lse4(zero_f, c_diag, c_top, c_left, inv_tau, tau, elt), "hij")
      if tot_var is not None:
        el.emit(Assign(tot_var, _lse2(tot_var, h_ij, inv_tau, tau, elt)))
    else:
      h_ij = el.tmp(_lse3(c_diag, c_top, c_left, inv_tau, tau, elt), "hij")
    el.emit(Store(cur_buf, iadd(cur_off, j), h_ij))
  el.emit(ForRange(j, jlo, jhi, ONE, jbody, jsc))


@libfunction("soft_align")
def soft_align(el, scores, gap_penalty, *, temperature=1.0, band=None, local=True):
  """Differentiable smooth log-semiring alignment (Soft-Smith-Waterman / Soft-Needleman-Wunsch).

  Uses O(N) rolling row memory in forward inference and **Law 2 checkpointed
  rematerialization** (O(sqrt(M) * N) memory, zero M x N tape) under ``pk.grad``.
  """
  sc = el.as_concrete(el.as_array(scores, "scores"))
  if sc.rank != 2 or sc.elt.kind != "f":
    raise el.error(TypeInferenceError, f"soft_align() expects a 2D float score matrix, got {describe(sc)}")
  elt = sc.elt
  gap_in = el.as_scalar(gap_penalty, "gap_penalty")
  gap = el.tmp(cast(gap_in, elt), "gap")
  tau = el.tmp(cast(el.as_scalar(temperature, "temperature"), elt), "tau")
  inv_tau = el.tmp(ops.prim("div", (Const(1.0, elt), tau), elt), "inv_tau")
  is_local = el.static_bool(local, "local") if not isinstance(local, bool) else local
  band_e = None if band is None else el.tmp(el.as_index(band, "band"), "band")
  M, N = sc.shape

  tape = el.tape
  record_ad = tape is not None and not tape.muted and (tape.is_active(el, sc) or tape.is_active(el, gap_in))
  saved_tape = el.tape
  if record_ad:
    el.tape = None
  try:
    # Chunk height B = max(1, int(sqrt(M))) for O(sqrt(M) * N) Law 2 checkpoints
    blk_b = el.tmp(imax(ONE, cast(ops.prim("sqrt", (cast(imax(M, ONE), Float64),), Float64), I)), "ckpt_B")
    num_ckpts = el.tmp(idiv(iadd(M, isub(blk_b, ONE)), blk_b), "nckpts")
    ckpts = (
      el.alloc_array(elt, (imax(num_ckpts, ONE), imax(N, ONE)), zero=True, label="sa_ckpts") if record_ad else None
    )

    row0 = el.alloc_array(elt, (imax(N, ONE),), zero=True, label="sa_r0")
    row1 = el.alloc_array(elt, (imax(N, ONE),), zero=True, label="sa_r1")
    out_var = Var("sa_out", elt)
    el.mutables.add(out_var.uid)
    el.emit(Assign(out_var, Const(-math.inf if is_local else 0.0, elt)))

    non_empty: list = []
    with el.in_block(non_empty):
      i = Var("i", I)
      el.nonneg.add(i.uid)
      isc = el.new_scope("loop")
      ibody: list = []
      with el.in_block(ibody, isc, loop=True):
        use_even = cmp("eq", ops.imod(i, Const(2, I)), ZERO)
        # Save checkpoint before every B-th row (row i - 1 lives in prev buffer)
        if ckpts is not None:
          is_ckpt = ops.land(cmp("gt", i, ZERO), cmp("eq", ops.imod(i, blk_b), ZERO))
          cblk: list = []
          with el.in_block(cblk):
            cidx = el.tmp(idiv(i, blk_b), "cidx")
            coff = el.tmp(imul(cidx, N), "coff")
            cj = Var("cj", I)
            el.nonneg.add(cj.uid)
            cjsc = el.new_scope("loop")
            cjb = [
              Store(
                ckpts.buf,
                iadd(coff, cj),
                ops.select(use_even, Load(row1.buf, cj), Load(row0.buf, cj)),
              )
            ]
            el.emit(ForRange(cj, ZERO, N, ONE, cjb, cjsc))
          el.emit(If(is_ckpt, cblk, []))

        even_blk: list = []
        with el.in_block(even_blk):
          _emit_soft_row_step(
            el,
            sc,
            gap,
            inv_tau,
            tau,
            elt,
            is_local,
            band_e,
            i,
            N,
            row1.buf,
            ZERO,
            row0.buf,
            ZERO,
            out_var if is_local else None,
          )
        odd_blk: list = []
        with el.in_block(odd_blk):
          _emit_soft_row_step(
            el,
            sc,
            gap,
            inv_tau,
            tau,
            elt,
            is_local,
            band_e,
            i,
            N,
            row0.buf,
            ZERO,
            row1.buf,
            ZERO,
            out_var if is_local else None,
          )
        el.emit(If(use_even, even_blk, odd_blk))
      el.emit(ForRange(i, ZERO, M, ONE, ibody, isc))
      if not is_local:
        last_even = cmp("eq", ops.imod(isub(M, ONE), Const(2, I)), ZERO)
        last_j = isub(N, ONE)
        last_val = ops.select(last_even, Load(row0.buf, last_j), Load(row1.buf, last_j))
        if band_e is not None:
          in_band = cmp("le", imax(isub(M, N), isub(N, M)), band_e)
          last_val = ops.select(in_band, last_val, Const(-math.inf, elt))
        el.emit(Assign(out_var, last_val))

    el.emit(If(ops.land(cmp("gt", M, ZERO), cmp("gt", N, ZERO)), non_empty, []))
  finally:
    el.tape = saved_tape

  if tape is not None and not tape.muted and tape.is_active(el, tau):
    raise el.error(UnsupportedError, "differentiating soft_align() with respect to temperature is not supported")
  if record_ad:
    from ..transforms.autodiff import Node

    node = Node(
      "soft_align",
      (sc, gap_in),
      (gap, tau, inv_tau, is_local, band_e, blk_b, num_ckpts, ckpts, out_var),
    )
    tape.record(el, node)
    tape.register(el, node, out_var)
  return out_var


def _soft_align_vjp(tape, el, node, g_out: Expr) -> None:
  """Law 2 checkpointed reverse-wavefront VJP for ``soft_align`` (O(sqrt(M) * N) memory)."""
  sc, gap_in = node.inputs
  gap, tau, inv_tau, is_local, band_e, blk_b, num_ckpts, ckpts, fwd_out = node.info
  tape.stale(el, node, sc)
  elt = sc.elt
  M, N = sc.shape
  g_s = el.tmp(cast(g_out, elt), "g_sa")
  zero_f = Const(0.0, elt)
  neginf = Const(-math.inf, elt)

  need_sc = tape.is_active(el, sc)
  need_gap = tape.is_active(el, gap_in)
  d_sc = el.alloc_array(elt, (M, N), zero=True, label="d_sa_sc") if need_sc else None
  d_gap = Var("d_sa_gap", elt)
  el.mutables.add(d_gap.uid)
  el.emit(Assign(d_gap, zero_f))

  # Local chunk buffer blk_H[(B + 1) * N] holds prev_row + B rematerialized rows
  blk_h = el.alloc_array(elt, (imul(iadd(blk_b, ONE), imax(N, ONE)),), zero=True, label="sa_blkh")
  dh_cur = el.alloc_array(elt, (imax(N, ONE),), zero=True, label="sa_dhc")
  dh_prev = el.alloc_array(elt, (imax(N, ONE),), zero=True, label="sa_dhp")

  non_empty: list = []
  with el.in_block(non_empty):
    c = Var("ck", I)
    csc = el.new_scope("loop")
    cbody: list = []
    with el.in_block(cbody, csc, loop=True):
      r0 = el.tmp(imul(c, blk_b), "r0")
      r1 = el.tmp(imin(iadd(r0, blk_b), M), "r1")
      rlen = el.tmp(isub(r1, r0), "rlen")
      # Load checkpoint row c into slot 0 of blk_h (only needed for c > 0)
      load_c: list = []
      with el.in_block(load_c):
        coff = el.tmp(imul(c, N), "coff")
        lj = Var("lj", I)
        el.nonneg.add(lj.uid)
        el.emit(ForRange(lj, ZERO, N, ONE, [Store(blk_h.buf, lj, Load(ckpts.buf, iadd(coff, lj)))], Scope("loop")))
      el.emit(If(cmp("gt", c, ZERO), load_c, []))

      # Rematerialize rows r0 .. r1-1 into blk_h slots 1 .. rlen
      rr = Var("rr", I)
      el.nonneg.add(rr.uid)
      rrsc = el.new_scope("loop")
      rrbody: list = []
      with el.in_block(rrbody, rrsc, loop=True):
        ri = el.tmp(iadd(r0, rr), "ri")
        poff = el.tmp(imul(rr, N), "poff")
        coff2 = el.tmp(imul(iadd(rr, ONE), N), "coff2")
        _emit_soft_row_step(
          el, sc, gap, inv_tau, tau, elt, is_local, band_e, ri, N, blk_h.buf, poff, blk_h.buf, coff2, None
        )
      el.emit(ForRange(rr, ZERO, rlen, ONE, rrbody, rrsc))

      # Reverse sweep rr = rlen - 1 down to 0
      br = Var("br", I)
      brsc = el.new_scope("loop")
      brbody: list = []
      with el.in_block(brbody, brsc, loop=True):
        i = el.tmp(iadd(r0, br), "bi")
        poff = el.tmp(imul(br, N), "bpoff")
        coff2 = el.tmp(imul(iadd(br, ONE), N), "bcoff")
        # Clear dh_prev
        zj = Var("zj", I)
        el.emit(ForRange(zj, ZERO, N, ONE, [Store(dh_prev.buf, zj, zero_f)], Scope("loop")))

        jlo = ZERO if band_e is None else el.tmp(imax(ZERO, isub(i, band_e)), "bjlo")
        jhi = isub(N, ONE) if band_e is None else el.tmp(imin(isub(N, ONE), iadd(i, band_e)), "bjhi")
        j = Var("bj", I)
        jsc = el.new_scope("loop")
        jbody: list = []
        with el.in_block(jbody, jsc, loop=True):
          h_ij = el.tmp(Load(blk_h.buf, iadd(coff2, j)), "bhij")
          g_cur = el.tmp(Load(dh_cur.buf, j), "bgc")
          if is_local:
            w_out = ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (h_ij, fwd_out), elt), inv_tau), elt),), elt)
            g_tot = el.tmp(ops.prim("add", (g_cur, ops.prim("mul", (g_s, w_out), elt)), elt), "bgt")
          else:
            is_end = ops.land(cmp("eq", i, isub(M, ONE)), cmp("eq", j, isub(N, ONE)))
            g_tot = el.tmp(ops.prim("add", (g_cur, ops.select(is_end, g_s, zero_f)), elt), "bgt")

          jm1 = el.tmp(imax(isub(j, ONE), ZERO), "bjm1")
          top_in_band = ops.TRUE if band_e is None else cmp("le", isub(j, isub(i, ONE)), band_e)
          left_in_band = ops.TRUE if band_e is None else cmp("le", isub(i, isub(j, ONE)), band_e)
          top_bnd = zero_f if is_local else ops.prim("mul", (cast(iadd(j, ONE), elt), gap), elt)
          left_bnd = zero_f if is_local else ops.prim("mul", (cast(iadd(i, ONE), elt), gap), elt)
          diag_bnd = (
            zero_f
            if is_local
            else ops.select(
              cmp("gt", j, ZERO),
              ops.prim("mul", (cast(j, elt), gap), elt),
              ops.select(cmp("gt", i, ZERO), ops.prim("mul", (cast(i, elt), gap), elt), zero_f),
            )
          )
          oob_val = zero_f if is_local else neginf

          top_v = el.tmp(
            ops.select(cmp("gt", i, ZERO), ops.select(top_in_band, Load(blk_h.buf, iadd(poff, j)), oob_val), top_bnd),
            "btv",
          )
          left_v = el.tmp(
            ops.select(
              cmp("gt", j, ZERO), ops.select(left_in_band, Load(blk_h.buf, iadd(coff2, jm1)), oob_val), left_bnd
            ),
            "blv",
          )
          diag_v = el.tmp(
            ops.select(ops.land(cmp("gt", i, ZERO), cmp("gt", j, ZERO)), Load(blk_h.buf, iadd(poff, jm1)), diag_bnd),
            "bdv",
          )

          c_diag = ops.prim("add", (diag_v, el.elem(sc, (i, j))), elt)
          c_top = ops.prim("add", (top_v, gap), elt)
          c_left = ops.prim("add", (left_v, gap), elt)

          w_diag = el.tmp(
            ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (c_diag, h_ij), elt), inv_tau), elt),), elt), "wd"
          )
          w_top = el.tmp(
            ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (c_top, h_ij), elt), inv_tau), elt),), elt), "wt"
          )
          w_left = el.tmp(
            ops.prim("exp", (ops.prim("mul", (ops.prim("sub", (c_left, h_ij), elt), inv_tau), elt),), elt), "wl"
          )

          gd = el.tmp(ops.prim("mul", (g_tot, w_diag), elt), "gd")
          gt = el.tmp(ops.prim("mul", (g_tot, w_top), elt), "gt")
          gl = el.tmp(ops.prim("mul", (g_tot, w_left), elt), "gl")

          if d_sc is not None:
            el.emit(Store(d_sc.buf, el.index_expr(d_sc, (i, j)), gd))
          if need_gap:
            dg_step = ops.prim("add", (gt, gl), elt)
            if not is_local:
              # Include boundary gap contributions at i == 0 and j == 0
              dg_step = ops.prim(
                "add",
                (
                  dg_step,
                  ops.select(cmp("eq", i, ZERO), ops.prim("mul", (gt, cast(iadd(j, ONE), elt)), elt), zero_f),
                ),
                elt,
              )
              dg_step = ops.prim(
                "add",
                (
                  dg_step,
                  ops.select(cmp("eq", j, ZERO), ops.prim("mul", (gl, cast(iadd(i, ONE), elt)), elt), zero_f),
                ),
                elt,
              )
              dg_diag_bnd = ops.select(
                ops.land(cmp("eq", i, ZERO), cmp("gt", j, ZERO)),
                ops.prim("mul", (gd, cast(j, elt)), elt),
                ops.select(
                  ops.land(cmp("eq", j, ZERO), cmp("gt", i, ZERO)),
                  ops.prim("mul", (gd, cast(i, elt)), elt),
                  zero_f,
                ),
              )
              dg_step = ops.prim("add", (dg_step, dg_diag_bnd), elt)
            el.emit(Assign(d_gap, ops.prim("add", (d_gap, dg_step), elt)))

          # Propagate adjoints to (i - 1, j), (i - 1, j - 1), and (i, j - 1)
          el.emit(
            If(
              ops.land(cmp("gt", i, ZERO), top_in_band),
              [Store(dh_prev.buf, j, ops.prim("add", (Load(dh_prev.buf, j), gt), elt))],
              [],
            )
          )
          el.emit(
            If(
              ops.land(cmp("gt", i, ZERO), cmp("gt", j, ZERO)),
              [Store(dh_prev.buf, jm1, ops.prim("add", (Load(dh_prev.buf, jm1), gd), elt))],
              [],
            )
          )
          el.emit(
            If(
              ops.land(cmp("gt", j, ZERO), left_in_band),
              [Store(dh_cur.buf, jm1, ops.prim("add", (Load(dh_cur.buf, jm1), gl), elt))],
              [],
            )
          )
        el.emit(ForRange(j, jhi, isub(jlo, ONE), Const(-1, I), jbody, jsc))

        # Copy dh_prev -> dh_cur for row i - 1
        cpj = Var("cpj", I)
        el.emit(ForRange(cpj, ZERO, N, ONE, [Store(dh_cur.buf, cpj, Load(dh_prev.buf, cpj))], Scope("loop")))
      el.emit(ForRange(br, isub(rlen, ONE), Const(-1, I), Const(-1, I), brbody, brsc))
    el.emit(ForRange(c, isub(num_ckpts, ONE), Const(-1, I), Const(-1, I), cbody, csc))

  el.emit(If(ops.land(cmp("gt", M, ZERO), cmp("gt", N, ZERO)), non_empty, []))
  if d_sc is not None:
    tape.add(el, sc, d_sc)
  if need_gap:
    tape.add(el, gap_in, cast(d_gap, gap_in.type))
