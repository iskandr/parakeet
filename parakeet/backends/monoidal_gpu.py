"""GPU SRAM-tiled code generator for :class:`~parakeet.syntax.ir.MonoidalTilePlan`.

Emits cooperative ``__shared__`` (CUDA) and ``threadgroup`` (Metal) kernels for
all four Law 1 / Law 2 / Law 5 rescaled-monoid plans:

* ``"flash_monoid"``: 1-pass FlashAttention / pairwise kernel smoothing with
  online ``(m, l, O)`` monoid and Law 5 tile-interval / boolean-ballot mask
  pruning (zero ``N x N`` HBM materialization).
* ``"flash_monoid_bwd"``: ``D_i``-recomputed FlashAttention backward (outer KV
  tiles keep ``dK`` and ``dV`` atomic-free in registers; ``dQ`` uses
  ``atomicAdd``) with Law 5 tile pruning.
* ``"monoid_lse"``: 1-pass Cut Cross-Entropy / pairwise LogSumExp forward with
  online ``(m, l)`` monoid and optional target gather (zero ``M x V`` HBM).
* ``"cce_bwd"``: SRAM-recomputed Cut Cross-Entropy backward (outer vocab tiles
  keep ``dW`` atomic-free in registers; ``dH`` uses ``atomicAdd``).

All generated kernels return ``extra={"tiled": ...}`` metadata compatible with
both ``CudaDevice`` / ``NativeCudaRuntime`` (``cuLaunchKernel``) and
``MetalDevice``.
"""

from __future__ import annotations

from ..ndtypes import BufType, Float32, ScalarType
from ..syntax import ir, ops
from ..syntax.ir import MonoidalTilePlan, ParLoop, TileStage, Var
from ..target import TargetProfile
from ..transforms.analysis import IndexFacts, TileInterval, tile_predicate_guards
from .cfamily import CFamilyEmitter, Unsupported, defined_vars
from .device import KernelPlan

MONOID_BM = 32
MONOID_BN = 32
MONOID_BK = 32
MONOID_THREADS = 128  # 32 rows x 4 threads/row (4 warps / SIMD-groups)
MAX_REG_D = 128  # 32 floats/thread * 4 threads/row = 128 columns in registers (clean fallback for larger D!)

# Hardware matrix-unit geometry (4 warps / SIMD-groups, 128 threads):
# - Metal simdgroup_matrix (8x8x8): BM=32, BN=32 (8 rows/SIMD-group, 4 lanes/row, 22 KiB SMEM, Q hoisted for K<=64!)
# - CUDA wmma TF32 (16x16x8):       BM=64, BN=32 (16 rows/warp, 2 lanes/row, 37 KiB SMEM, Q hoisted for K<=64!)
MM_BM = 64
MM_BN = 32
MM_BK = 32
MM_THREADS = 128
MM_METAL_BM = 32
MM_METAL_BN = 32
MM_METAL_BK = 32


def _emit_decl_locals(em: CFamilyEmitter, stmts: list[ir.Stmt], skip_uids: set[int], out: list[str], pad: str) -> None:
  defs = defined_vars(stmts, into_parloops=True)
  for uid, v in sorted(defs.items()):
    if uid in skip_uids:
      continue
    if isinstance(v.type, BufType):
      raise Unsupported("monoidal GPU tile allocates a buffer inside a stage")
    init = "false" if v.type.kind == "b" else "0"
    out.append(f"{pad}{em.ctype(v.type)} {v.name} = {init};")


def _stage_prefers_dim0_fast(st: TileStage) -> bool:
  """True when staging ``st`` coalesces better with dim 0 fastest."""
  idx0_uid = st.index[0].uid
  idx1_uid = st.index[1].uid
  d0_unit = False
  d1_unit = False
  for expr in (st.value, *(e for s in st.body for e in ir.stmt_exprs(s))):
    for x in ir.walk_expr(expr):
      if isinstance(x, ir.Load):
        idx = x.index
        if isinstance(idx, ir.Prim) and idx.op == "add":
          for a in idx.args:
            if isinstance(a, Var) and a.uid == idx0_uid:
              d0_unit = True
            if isinstance(a, Var) and a.uid == idx1_uid:
              d1_unit = True
  return d0_unit and not d1_unit


def _emit_stage_load_2d(
  em: CFamilyEmitter,
  st: TileStage,
  sh_name: str,
  r_size: int,
  c_size: int,
  c_stride: int,
  r0_expr: str,
  c0_expr: str,
  r_lim: str,
  c_lim: str,
  ct: str,
  idx_t: str,
  lines: list[str],
  pad: str,
  zero_lit: str,
) -> None:
  """Cooperatively load a ``r_size x c_size`` tile of ``st`` into ``sh_name[r * c_stride + c]``."""
  r_shift = r_size.bit_length() - 1
  c_shift = c_size.bit_length() - 1
  d0_fast = _stage_prefers_dim0_fast(st)
  lines.append(f"{pad}for ({idx_t} pk_l = pk_tid; pk_l < {r_size * c_size}; pk_l += {MONOID_THREADS}) {{")
  if d0_fast:
    lines.append(f"{pad}  const {idx_t} pk_lc = pk_l >> {r_shift};")
    lines.append(f"{pad}  const {idx_t} pk_lr = pk_l & {r_size - 1};")
  else:
    lines.append(f"{pad}  const {idx_t} pk_lr = pk_l >> {c_shift};")
    lines.append(f"{pad}  const {idx_t} pk_lc = pk_l & {c_size - 1};")
  lines.append(f"{pad}  const {idx_t} {st.index[0].name} = {r0_expr} + pk_lr;")
  lines.append(f"{pad}  const {idx_t} {st.index[1].name} = {c0_expr} + pk_lc;")
  lines.append(f"{pad}  if ({st.index[0].name} < {r_lim} && {st.index[1].name} < {c_lim}) {{")
  _emit_decl_locals(em, st.body, {st.index[0].uid, st.index[1].uid}, lines, pad + "    ")
  em.block(st.body, lines, (len(pad) + 4) // 2)
  lines.append(f"{pad}    {sh_name}[pk_lr * {c_stride} + pk_lc] = ({ct})({em.expr(st.value)});")
  lines.append(f"{pad}  }} else {{")
  lines.append(f"{pad}    {sh_name}[pk_lr * {c_stride} + pk_lc] = {zero_lit};")
  lines.append(f"{pad}  }}")
  lines.append(f"{pad}}}")


def _emit_inline_stage_2d(
  em: CFamilyEmitter,
  st: TileStage,
  r_expr: str,
  c_expr: str,
  target_var: str,
  ct: str,
  idx_t: str,
  lines: list[str],
  pad: str,
) -> None:
  """Evaluate 2D ``TileStage`` ``st`` inline at ``(r_expr, c_expr)`` into ``target_var``."""
  lines.append(f"{pad}{ct} {target_var};")
  lines.append(f"{pad}{{")
  lines.append(f"{pad}  const {idx_t} {st.index[0].name} = {r_expr};")
  lines.append(f"{pad}  const {idx_t} {st.index[1].name} = {c_expr};")
  _emit_decl_locals(em, st.body, {st.index[0].uid, st.index[1].uid}, lines, pad + "  ")
  em.block(st.body, lines, (len(pad) + 2) // 2)
  lines.append(f"{pad}  {target_var} = ({ct})({em.expr(st.value)});")
  lines.append(f"{pad}}}")


def _emit_header(
  em: CFamilyEmitter,
  target: str,
  prelude: str,
  scalars: list[Var],
  bufs: list[Var],
  need_atomic_f32: bool = False,
) -> tuple[list[str], str, str, str, str]:
  is_cuda = target == "cuda"
  idx_t = "long long" if is_cuda else "long"
  sh_qual = "__shared__" if is_cuda else "threadgroup"
  barrier = "__syncthreads();" if is_cuda else "threadgroup_barrier(mem_flags::mem_threadgroup);"
  lines: list[str] = [prelude]
  if not is_cuda and need_atomic_f32:
    lines.append(
      """static inline void pk_atomic_add_f32(device float* addr, float val) {
  device atomic_uint* uaddr = (device atomic_uint*)addr;
  uint old_u = atomic_load_explicit(uaddr, memory_order_relaxed);
  while (true) {
    uint new_u = as_type<uint>(as_type<float>(old_u) + val);
    if (atomic_compare_exchange_weak_explicit(uaddr, &old_u, new_u, memory_order_relaxed, memory_order_relaxed)) break;
  }
}"""
    )
  if is_cuda:
    params = ["long long pk_total", "long long pk_csize"]
    params += [f"{em.ctype(v.type)} {v.name}" for v in scalars]
    params += ["int* pk_flag", "long long* pk_info"]
    params += [f"{em.ctype(v.type.elt)}* {v.name}_d" for v in bufs]
    lines.append('extern "C" __global__ void pk_kernel(' + ",\n    ".join(params) + ") {")
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})blockIdx.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})threadIdx.x;")
  else:
    lines += ["struct pk_args {", "  long total;", "  long csize;"]
    for v in scalars:
      lines.append(f"  {em.ctype(v.type)} {v.name};")
    lines.append("};")
    params = [
      "constant pk_args& pk_a [[buffer(0)]]",
      "device atomic_int* pk_flag [[buffer(1)]]",
      "device long* pk_info [[buffer(2)]]",
    ]
    for k, v in enumerate(bufs):
      params.append(f"device {em.ctype(v.type.elt)}* {v.name}_d [[buffer({k + 3})]]")
    params.append("uint3 pk_bid [[threadgroup_position_in_grid]]")
    params.append("uint3 pk_ltid [[thread_position_in_threadgroup]]")
    lines.append("kernel void pk_kernel(" + ",\n    ".join(params) + ") {")
    for v in scalars:
      lines.append(f"  const {em.ctype(v.type)} {v.name} = pk_a.{v.name};")
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})pk_bid.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})pk_ltid.x;")
  lines.append(f"  const {idx_t} pk_r = pk_tid >> 2;")  # row in [0, 32)
  lines.append(f"  const {idx_t} pk_t = pk_tid & 3;")  # 4 lanes per row
  return lines, idx_t, sh_qual, barrier, "0LL" if is_cuda else "0L"


def _emit_batch_unravel(
  em: CFamilyEmitter,
  plan: MonoidalTilePlan,
  row_dim_var: str,
  bm: int,
  idx_t: str,
  lines: list[str],
) -> None:
  """Unravel ``pk_blk`` into ``plan.batch_index`` and ``pk_blk_r`` (tile index along ``row_dim_var``)."""
  lines.append(f"  const {idx_t} pk_nb_r = ({row_dim_var} + {bm - 1}) / {bm};")
  lines.append(f"  const {idx_t} pk_blk_r = pk_blk % pk_nb_r;")
  if plan.batch_shape:
    lines.append(f"  {idx_t} pk_bflat = pk_blk / pk_nb_r;")
    for b_idx in range(len(plan.batch_shape) - 1, 0, -1):
      d_e = em.expr(plan.batch_shape[b_idx])
      bv = plan.batch_index[b_idx]
      lines.append(f"  const {idx_t} {bv.name} = pk_bflat % ({idx_t})({d_e});")
      lines.append(f"  pk_bflat = pk_bflat / ({idx_t})({d_e});")
    lines.append(f"  const {idx_t} {plan.batch_index[0].name} = pk_bflat;")
    # Flat batch index
    lines.append(f"  const {idx_t} pk_batch_id = pk_blk / pk_nb_r;")
  else:
    lines.append(f"  const {idx_t} pk_batch_id = 0;")


def _emit_atomic_add(is_cuda: bool, ptr_expr: str, val_expr: str, elt: ScalarType, lines: list[str], pad: str) -> None:
  if is_cuda:
    lines.append(f"{pad}atomicAdd({ptr_expr}, {val_expr});")
  else:
    if elt.name != "float32":
      raise Unsupported("Metal monoidal backward atomics require float32")
    lines.append(f"{pad}pk_atomic_add_f32({ptr_expr}, {val_expr});")


def _lit(val: float, elt: ScalarType, is_cuda: bool) -> str:
  if val == float("-inf"):
    return "(-INFINITY)" if not is_cuda else ("(-1e38f)" if elt.name == "float32" else "(-1e300)")
  s = repr(float(val))
  return s + "f" if (elt.name == "float32" and is_cuda) else s


def _exp_fn(elt: ScalarType, is_cuda: bool) -> str:
  return "expf" if (elt.name == "float32" and is_cuda) else "exp"


def _log_fn(elt: ScalarType, is_cuda: bool) -> str:
  return "logf" if (elt.name == "float32" and is_cuda) else "log"


def _max_fn(elt: ScalarType, is_cuda: bool) -> str:
  return "fmaxf" if (elt.name == "float32" and is_cuda) else ("fmax" if is_cuda else "fmax")


def _generate_flash_monoid(
  loop: ParLoop,
  plan: MonoidalTilePlan,
  free: list[Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None,
) -> KernelPlan:
  """Generate 1-pass SRAM-tiled FlashAttention / pairwise kernel smoothing GPU kernel."""
  is_cuda = target == "cuda"
  elt = plan.elt
  ct = em.ctype(elt)
  zero_lit = _lit(0.0, elt, is_cuda)
  neg_inf = _lit(float("-inf"), elt, is_cuda)
  expf = _exp_fn(elt, is_cuda)
  logf = _log_fn(elt, is_cuda)
  fmax = _max_fn(elt, is_cuda)

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  lines, idx_t, sh_qual, barrier, _ = _emit_header(em, target, prelude, scalars, bufs)
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.m_dim)});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.n_dim)});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.k_dim)});")
  assert plan.d_dim is not None
  lines.append(f"  const {idx_t} pk_D = ({idx_t})({em.expr(plan.d_dim)});")
  _emit_batch_unravel(em, plan, "pk_M", MONOID_BM, idx_t, lines)
  lines.append(f"  const {idx_t} pk_i0 = pk_blk_r * {MONOID_BM};")
  lines.append(f"  const {idx_t} pk_gi = pk_i0 + pk_r;")
  lines.append(f"  const {idx_t} pk_row_flat = pk_batch_id * pk_M + pk_gi;")
  if plan.score_scale is not None:
    lines.append(f"  const {ct} pk_scale = ({ct})({em.expr(plan.score_scale)});")
  has_ipa = (
    "xq" in plan.stages and "xk" in plan.stages and "ipa_gamma" in plan.aux_exprs and "ipa_c_dim" in plan.aux_exprs
  )
  if has_ipa:
    lines.append(f"  const {ct} pk_ipa_gamma = ({ct})({em.expr(plan.aux_exprs['ipa_gamma'])});")
    lines.append(f"  const {idx_t} pk_C = ({idx_t})({em.expr(plan.aux_exprs['ipa_c_dim'])});")
  has_bias = "bias" in plan.stages
  if plan.mask_fill is not None:
    lines.append(f"  const {ct} pk_mfill = ({ct})({em.expr(plan.mask_fill)});")

  # Shared memory tiles: sh_q[32x65] (hoisted once when K<=64!), sh_b[32x65], sh_s[32x33]
  s_pad = MONOID_BN + 1
  k_pad = MONOID_BK + 1
  q_fast_pad = 65
  lines.append(f"  {sh_qual} {ct} pk_sh_a[{MONOID_BM * q_fast_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_b[{MONOID_BN * q_fast_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_s[{MONOID_BM * s_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_m[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_l[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_alpha[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_red[{MONOID_BM * 4}];")
  has_mask_buf = plan.mask_buf is not None and plan.mask_index is not None and plan.mask_fill is not None
  has_mask_cond = plan.mask_cond is not None and plan.mask_fill is not None
  if has_mask_buf:
    lines.append(f"  {sh_qual} unsigned char pk_sh_mask[{MONOID_BM * MONOID_BN}];")
    lines.append(f"  {sh_qual} int pk_any_mask;")
    lines.append(f"  {sh_qual} int pk_row_valid[{MONOID_BM}];")
    lines.append(f"  {sh_qual} int pk_all_valid;")
    lines.append("  if (pk_tid < 32) pk_row_valid[pk_tid] = 0;")
    lines.append("  if (pk_tid == 0) pk_all_valid = 0;")

  # Initialize per-row monoid state (m = -inf, l = 0) and O output buffer to 0
  o_buf = plan.out_bufs["o"]
  lines.append("  if (pk_t == 0) {")
  lines.append(f"    pk_sh_m[pk_r] = {neg_inf};")
  lines.append(f"    pk_sh_l[pk_r] = {zero_lit};")
  lines.append("  }")
  lines.append(f"  {ct} pk_o_reg[{MAX_REG_D // 4}];")
  lines.append(f"  for (int pk_u = 0; pk_u < {MAX_REG_D // 4}; ++pk_u) pk_o_reg[pk_u] = {zero_lit};")
  lines.append("  if (pk_gi < pk_M && pk_D > 128) {")
  lines.append(f"    for ({idx_t} pk_d = pk_t + {MAX_REG_D}; pk_d < pk_D; pk_d += 4) {{")
  lines.append(f"      {em.data(o_buf)}[pk_row_flat * pk_D + pk_d] = {zero_lit};")
  lines.append("    }")
  lines.append("  }")
  # Hoist Q_tile[32 x K] ONCE into pk_sh_a when K <= 64 (eliminates re-reading Q on every KV tile!)
  lines.append("  const bool pk_q_hoisted = (pk_K <= 64);")
  lines.append("  if (pk_q_hoisted) {")
  _emit_stage_load_2d(
    em,
    plan.stages["q"],
    "pk_sh_a",
    MONOID_BM,
    64,
    q_fast_pad,
    "pk_i0",
    "0",
    "pk_M",
    "pk_K",
    ct,
    idx_t,
    lines,
    "    ",
    zero_lit,
  )
  lines.append("  }")
  lines.append(f"  {barrier}")

  # Main outer loop over KV tiles j0 in [pk_j_start, pk_j_end) (supports FlashDecoding Split-K!)
  if "j_start" in plan.aux_exprs and "j_end" in plan.aux_exprs:
    lines.append(f"  const {idx_t} pk_j_start = ({idx_t})({em.expr(plan.aux_exprs['j_start'])});")
    lines.append(f"  const {idx_t} pk_j_end = ({idx_t})({em.expr(plan.aux_exprs['j_end'])});")
  else:
    lines.append(f"  const {idx_t} pk_j_start = 0;")
    lines.append(f"  const {idx_t} pk_j_end = pk_N;")
  is_neg_inf_fill = isinstance(plan.mask_fill, ir.Const) and float(plan.mask_fill.value) == float("-inf")
  skip_eps = _lit(105.0 if elt.bits <= 32 else 750.0, elt, is_cuda)

  lines.append(f"  for ({idx_t} pk_j0 = pk_j_start; pk_j0 < pk_j_end; pk_j0 += {MONOID_BN}) {{")
  # Law 5 symbolic tile skip: only skip when all-false tile contributes the exact reduction identity
  # (mask_fill == -inf, or every active row already has l > 0 and m > mask_fill + skip_eps so exp(fill - m) == 0).
  if plan.tile_false_guard is not None and plan.tile_origin_vars is not None:
    i0_v, j0_v = plan.tile_origin_vars
    lines.append(f"    const {idx_t} {i0_v.name} = pk_i0;")
    lines.append(f"    const {idx_t} {j0_v.name} = pk_j0;")
    if is_neg_inf_fill:
      lines.append(f"    if ({em.expr(plan.tile_false_guard)}) continue;")
    else:
      lines.append(f"    if ({em.expr(plan.tile_false_guard)}) {{")
      lines.append("      int pk_can_skip = 1;")
      lines.append(f"      for ({idx_t} pk_ri = 0; pk_ri < {MONOID_BM}; ++pk_ri) {{")
      lines.append(
        f"        if (pk_i0 + pk_ri < pk_M && !(pk_sh_l[pk_ri] > {zero_lit} && pk_sh_m[pk_ri] > pk_mfill + {skip_eps})) {{ pk_can_skip = 0; break; }}"
      )
      lines.append("      }")
      lines.append("      if (pk_can_skip) continue;")
      lines.append("    }")
  if plan.tile_true_guard is not None and plan.tile_origin_vars is not None:
    i0_v, j0_v = plan.tile_origin_vars
    if plan.tile_false_guard is None:
      lines.append(f"    const {idx_t} {i0_v.name} = pk_i0;")
      lines.append(f"    const {idx_t} {j0_v.name} = pk_j0;")
    lines.append(f"    const bool pk_tile_all_true = ({em.expr(plan.tile_true_guard)});")
  else:
    lines.append("    const bool pk_tile_all_true = false;")

  # Law 5 boolean-mask ballot skip (gated on the same reduction-identity condition!)
  if has_mask_buf:
    assert plan.mask_index_vars is not None and plan.mask_buf is not None and plan.mask_index is not None
    mi_v, mj_v = plan.mask_index_vars
    lines.append("    if (pk_tid == 0) pk_any_mask = 0;")
    lines.append(f"    {barrier}")
    lines.append(f"    for ({idx_t} pk_ml = pk_tid; pk_ml < {MONOID_BM * MONOID_BN}; pk_ml += {MONOID_THREADS}) {{")
    lines.append("      const " + idx_t + " pk_mr = pk_ml >> 5;")
    lines.append("      const " + idx_t + " pk_mc = pk_ml & 31;")
    lines.append(f"      const {idx_t} {mi_v.name} = pk_i0 + pk_mr;")
    lines.append(f"      const {idx_t} {mj_v.name} = pk_j0 + pk_mc;")
    lines.append("      unsigned char pk_mv = 0;")
    lines.append(f"      if ({mi_v.name} < pk_M && {mj_v.name} < pk_N) {{")
    lines.append(f"        pk_mv = {em.data(plan.mask_buf)}[{em.expr(plan.mask_index)}] ? 1 : 0;")
    lines.append("        if (pk_mv) { pk_any_mask = 1; pk_row_valid[pk_mr] = 1; }")
    lines.append("      }")
    lines.append("      pk_sh_mask[pk_ml] = pk_mv;")
    lines.append("    }")
    lines.append(f"    {barrier}")
    if is_neg_inf_fill:
      lines.append("    if (pk_any_mask == 0 && pk_all_valid) continue;")
    else:
      lines.append("    if (pk_any_mask == 0 && pk_all_valid) {")
      lines.append("      int pk_can_skip = 1;")
      lines.append(f"      for ({idx_t} pk_ri = 0; pk_ri < {MONOID_BM}; ++pk_ri) {{")
      lines.append(
        f"        if (pk_i0 + pk_ri < pk_M && !(pk_sh_l[pk_ri] > {zero_lit} && pk_sh_m[pk_ri] > pk_mfill + {skip_eps})) {{ pk_can_skip = 0; break; }}"
      )
      lines.append("      }")
      lines.append("      if (pk_can_skip) continue;")
      lines.append("    }")
    lines.append("    if (pk_tid == 0 && !pk_all_valid) {")
    lines.append("      int pk_av = 1;")
    lines.append(f"      for ({idx_t} pk_ri = 0; pk_ri < {MONOID_BM}; ++pk_ri) {{")
    lines.append("        if (pk_i0 + pk_ri < pk_M && !pk_row_valid[pk_ri]) { pk_av = 0; break; }")
    lines.append("      }")
    lines.append("      pk_all_valid = pk_av;")
    lines.append("    }")

  # Compute S_tile[32][32] = Q_tile @ K_tile^T (fast hoisted-Q path for K<=64, chunked BK=32 for arbitrary K)
  lines.append(f"    {ct} pk_s_acc[8];")
  lines.append(f"    for (int pk_c = 0; pk_c < 8; ++pk_c) pk_s_acc[pk_c] = {zero_lit};")
  lines.append("    if (pk_q_hoisted) {")
  _emit_stage_load_2d(
    em,
    plan.stages["k"],
    "pk_sh_b",
    MONOID_BN,
    64,
    q_fast_pad,
    "pk_j0",
    "0",
    "pk_j_end",
    "pk_K",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  lines.append(f"      {barrier}")
  lines.append("      #pragma unroll")
  lines.append("      for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("        const " + idx_t + " pk_jc = pk_c * 4 + pk_t;")
  lines.append(f"        {ct} pk_dot = {zero_lit};")
  lines.append("        for (" + idx_t + " pk_kk = 0; pk_kk < pk_K; ++pk_kk) {")
  if plan.score_op == "sqdist":
    lines.append(
      f"          {ct} pk_df = pk_sh_a[pk_r * {q_fast_pad} + pk_kk] - pk_sh_b[pk_jc * {q_fast_pad} + pk_kk];"
    )
    lines.append("          pk_dot += pk_df * pk_df;")
  else:
    lines.append(f"          pk_dot += pk_sh_a[pk_r * {q_fast_pad} + pk_kk] * pk_sh_b[pk_jc * {q_fast_pad} + pk_kk];")
  lines.append("        }")
  lines.append("        pk_s_acc[pk_c] = pk_dot;")
  lines.append("      }")
  lines.append(f"      {barrier}")
  lines.append("    } else {")
  lines.append(f"      for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {MONOID_BK}) {{")
  _emit_stage_load_2d(
    em,
    plan.stages["q"],
    "pk_sh_a",
    MONOID_BM,
    MONOID_BK,
    k_pad,
    "pk_i0",
    "pk_k0",
    "pk_M",
    "pk_K",
    ct,
    idx_t,
    lines,
    "        ",
    zero_lit,
  )
  _emit_stage_load_2d(
    em,
    plan.stages["k"],
    "pk_sh_b",
    MONOID_BN,
    MONOID_BK,
    k_pad,
    "pk_j0",
    "pk_k0",
    "pk_j_end",
    "pk_K",
    ct,
    idx_t,
    lines,
    "        ",
    zero_lit,
  )
  lines.append(f"        {barrier}")
  lines.append(f"        const {idx_t} pk_klen = pk_K - pk_k0 < {MONOID_BK} ? pk_K - pk_k0 : {MONOID_BK};")
  lines.append("        for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("          const " + idx_t + " pk_jc = pk_c * 4 + pk_t;")
  lines.append(f"          {ct} pk_dot = pk_s_acc[pk_c];")
  lines.append(f"          for ({idx_t} pk_kk = 0; pk_kk < pk_klen; ++pk_kk) {{")
  if plan.score_op == "sqdist":
    lines.append(f"            {ct} pk_df = pk_sh_a[pk_r * {k_pad} + pk_kk] - pk_sh_b[pk_jc * {k_pad} + pk_kk];")
    lines.append("            pk_dot += pk_df * pk_df;")
  else:
    lines.append(f"            pk_dot += pk_sh_a[pk_r * {k_pad} + pk_kk] * pk_sh_b[pk_jc * {k_pad} + pk_kk];")
  lines.append("          }")
  lines.append("          pk_s_acc[pk_c] = pk_dot;")
  lines.append("        }")
  lines.append(f"        {barrier}")
  lines.append("      }")
  lines.append("    }")

  # Apply scale + mask and compute local row max across this thread's 8 columns
  lines.append(f"    {ct} pk_lmax = {neg_inf};")
  lines.append("    for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("      const " + idx_t + " pk_jc = pk_c * 4 + pk_t;")
  lines.append("      const " + idx_t + " pk_gj = pk_j0 + pk_jc;")
  lines.append(f"      {ct} pk_sv = {neg_inf};")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_j_end) {")
  lines.append("        pk_sv = pk_s_acc[pk_c];")
  if plan.score_scale is not None:
    lines.append("        pk_sv *= pk_scale;")
  if has_ipa:
    lines.append(f"        {ct} pk_sq = {zero_lit};")
    lines.append(f"        for ({idx_t} pk_cc = 0; pk_cc < pk_C; ++pk_cc) {{")
    _emit_inline_stage_2d(em, plan.stages["xq"], "pk_gi", "pk_cc", "pk_xqv", ct, idx_t, lines, "          ")
    _emit_inline_stage_2d(em, plan.stages["xk"], "pk_gj", "pk_cc", "pk_xkv", ct, idx_t, lines, "          ")
    lines.append(f"          const {ct} pk_df = pk_xqv - pk_xkv;")
    lines.append("          pk_sq += pk_df * pk_df;")
    lines.append("        }")
    lines.append("        pk_sv -= pk_ipa_gamma * pk_sq;")
  if has_bias:
    _emit_inline_stage_2d(em, plan.stages["bias"], "pk_gi", "pk_gj", "pk_bv", ct, idx_t, lines, "        ")
    lines.append("        pk_sv += pk_bv;")
  if has_mask_buf:
    lines.append("        if (!pk_sh_mask[pk_r * 32 + pk_jc]) pk_sv = pk_mfill;")
  elif has_mask_cond:
    assert plan.mask_index_vars is not None and plan.mask_cond is not None
    mi_v, mj_v = plan.mask_index_vars
    lines.append("        if (!pk_tile_all_true) {")
    lines.append(f"          const {idx_t} {mi_v.name} = pk_gi;")
    lines.append(f"          const {idx_t} {mj_v.name} = pk_gj;")
    lines.append(f"          if (!({em.expr(plan.mask_cond)})) pk_sv = pk_mfill;")
    lines.append("        }")
  lines.append(f"        pk_lmax = {fmax}(pk_lmax, pk_sv);")
  lines.append("      }")
  lines.append(f"      pk_sh_s[pk_r * {s_pad} + pk_jc] = pk_sv;")
  lines.append("    }")
  lines.append("    pk_sh_red[pk_r * 4 + pk_t] = pk_lmax;")
  lines.append(f"    {barrier}")

  # Reduce row max across 4 threads -> compute m_new and alpha
  lines.append("    if (pk_t == 0 && pk_gi < pk_M) {")
  lines.append(
    f"      {ct} pk_tm = {fmax}({fmax}(pk_sh_red[pk_r*4], pk_sh_red[pk_r*4+1]), {fmax}(pk_sh_red[pk_r*4+2], pk_sh_red[pk_r*4+3]));"
  )
  lines.append("      if (pk_j0 == pk_j_start) {")
  lines.append("        pk_sh_m[pk_r] = pk_tm;")
  lines.append(f"        pk_sh_alpha[pk_r] = {zero_lit};")
  lines.append("      } else {")
  lines.append(f"        {ct} pk_mold = pk_sh_m[pk_r];")
  lines.append(f"        {ct} pk_mnew = {fmax}(pk_mold, pk_tm);")
  lines.append("        pk_sh_m[pk_r] = pk_mnew;")
  lines.append(f"        pk_sh_alpha[pk_r] = {expf}(pk_mold - pk_mnew);")
  lines.append("      }")
  lines.append("    }")
  lines.append(f"    {barrier}")

  # Compute P_tile = exp(S_tile - m_new) and row sum
  lines.append(f"    const {ct} pk_mcur = pk_sh_m[pk_r];")
  lines.append(f"    const {ct} pk_al = pk_sh_alpha[pk_r];")
  lines.append(f"    {ct} pk_lsum = {zero_lit};")
  lines.append("    for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("      const " + idx_t + " pk_jc = pk_c * 4 + pk_t;")
  lines.append("      const " + idx_t + " pk_gj = pk_j0 + pk_jc;")
  lines.append(f"      {ct} pk_pv = {zero_lit};")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_j_end) {")
  lines.append(f"        pk_pv = {expf}(pk_sh_s[pk_r * {s_pad} + pk_jc] - pk_mcur);")
  lines.append("        pk_lsum += pk_pv;")
  lines.append("      }")
  lines.append(f"      pk_sh_s[pk_r * {s_pad} + pk_jc] = pk_pv;")
  lines.append("    }")
  lines.append("    pk_sh_red[pk_r * 4 + pk_t] = pk_lsum;")
  lines.append(f"    {barrier}")
  lines.append("    if (pk_t == 0 && pk_gi < pk_M) {")
  lines.append(
    "      pk_sh_l[pk_r] = pk_sh_l[pk_r] * pk_al + (pk_sh_red[pk_r*4] + pk_sh_red[pk_r*4+1] + pk_sh_red[pk_r*4+2] + pk_sh_red[pk_r*4+3]);"
  )
  lines.append("    }")

  # Accumulate O = alpha * O + P_tile @ V_tile (in chunks of BD=32; first 128 cols in registers!)
  lines.append(f"    const {idx_t} pk_jlen = pk_j_end - pk_j0 < {MONOID_BN} ? pk_j_end - pk_j0 : {MONOID_BN};")
  lines.append(f"    for ({idx_t} pk_d0 = 0; pk_d0 < pk_D; pk_d0 += {MONOID_BK}) {{")
  _emit_stage_load_2d(
    em,
    plan.stages["v"],
    "pk_sh_b",
    MONOID_BN,
    MONOID_BK,
    k_pad,
    "pk_j0",
    "pk_d0",
    "pk_j_end",
    "pk_D",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  lines.append(f"      {barrier}")
  lines.append("      for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("        const " + idx_t + " pk_dc = pk_c * 4 + pk_t;")
  lines.append("        const " + idx_t + " pk_gd = pk_d0 + pk_dc;")
  lines.append("        if (pk_gi < pk_M && pk_gd < pk_D) {")
  lines.append(f"          {ct} pk_pv_sum = {zero_lit};")
  lines.append(f"          for ({idx_t} pk_jj = 0; pk_jj < pk_jlen; ++pk_jj) {{")
  lines.append(f"            pk_pv_sum += pk_sh_s[pk_r * {s_pad} + pk_jj] * pk_sh_b[pk_jj * {k_pad} + pk_dc];")
  lines.append("          }")
  lines.append(f"          if (pk_d0 < {MAX_REG_D}) {{")
  lines.append("            const int pk_slot = (int)((pk_d0 >> 2) + pk_c);")
  lines.append("            pk_o_reg[pk_slot] = pk_o_reg[pk_slot] * pk_al + pk_pv_sum;")
  lines.append("          } else {")
  lines.append(
    f"            {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] = {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] * pk_al + pk_pv_sum;"
  )
  lines.append("          }")
  lines.append("        }")
  lines.append("      }")
  lines.append(f"      {barrier}")
  lines.append("    }")
  lines.append("  }")  # end j0 loop

  # Final normalize O /= l (or keep unnormalized for FlashDecoding Split-K partial pass)
  unnorm = "unnormalized" in plan.aux_exprs
  lines.append("  if (pk_gi < pk_M) {")
  if unnorm:
    lines.append(f"    const {ct} pk_inv_l = {_lit(1.0, elt, is_cuda)};")
  else:
    lines.append(f"    const {ct} pk_inv_l = {_lit(1.0, elt, is_cuda)} / pk_sh_l[pk_r];")
  lines.append(f"    for (int pk_u = 0; pk_u < {MAX_REG_D // 4}; ++pk_u) {{")
  lines.append("      const " + idx_t + " pk_gd = pk_u * 4 + pk_t;")
  lines.append("      if (pk_gd < pk_D) {")
  lines.append(f"        {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] = pk_o_reg[pk_u] * pk_inv_l;")
  lines.append("      }")
  lines.append("    }")
  lines.append(f"    for ({idx_t} pk_d = pk_t + {MAX_REG_D}; pk_d < pk_D; pk_d += 4) {{")
  lines.append(f"      {em.data(o_buf)}[pk_row_flat * pk_D + pk_d] *= pk_inv_l;")
  lines.append("    }")
  lines.append("    if (pk_t == 0) {")
  if "m" in plan.out_bufs:
    lines.append(f"      {em.data(plan.out_bufs['m'])}[pk_row_flat] = pk_sh_m[pk_r];")
  if "l" in plan.out_bufs:
    lines.append(f"      {em.data(plan.out_bufs['l'])}[pk_row_flat] = pk_sh_l[pk_r];")
  if "lse" in plan.out_bufs:
    lines.append(f"      {em.data(plan.out_bufs['lse'])}[pk_row_flat] = pk_sh_m[pk_r] + {logf}(pk_sh_l[pk_r]);")
  lines.append("    }")
  lines.append("  }")
  lines.append("}")

  written = {b.uid for b in plan.out_bufs.values()}
  grid_shape = [*plan.batch_shape, plan.m_dim, ir.Const(1, ops.I)]
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in grid_shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={
      "tiled": {
        "bm": MONOID_BM,
        "bn": 1,
        "bk": MONOID_BK,
        "tm": 1,
        "tn": 1,
        "threads": MONOID_THREADS,
        "shape": shape_meta,
      }
    },
  )


def _generate_flash_monoid_bwd(
  loop: ParLoop,
  plan: MonoidalTilePlan,
  free: list[Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None,
) -> KernelPlan:
  """Generate D_i-recomputed FlashAttention backward GPU kernel (outer KV tiles)."""
  is_cuda = target == "cuda"
  elt = plan.elt
  ct = em.ctype(elt)
  zero_lit = _lit(0.0, elt, is_cuda)
  expf = _exp_fn(elt, is_cuda)

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  lines, idx_t, sh_qual, barrier, _ = _emit_header(em, target, prelude, scalars, bufs, need_atomic_f32=True)
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.m_dim)});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.n_dim)});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.k_dim)});")
  assert plan.d_dim is not None
  lines.append(f"  const {idx_t} pk_D = ({idx_t})({em.expr(plan.d_dim)});")
  _emit_batch_unravel(em, plan, "pk_N", MONOID_BN, idx_t, lines)
  lines.append(f"  const {idx_t} pk_j0 = pk_blk_r * {MONOID_BN};")
  lines.append(f"  const {idx_t} pk_gj = pk_j0 + pk_r;")
  lines.append(f"  const {idx_t} pk_jrow_flat = pk_batch_id * pk_N + pk_gj;")
  if plan.score_scale is not None:
    lines.append(f"  const {ct} pk_scale = ({ct})({em.expr(plan.score_scale)});")
  has_ipa = (
    "xq" in plan.stages and "xk" in plan.stages and "ipa_gamma" in plan.aux_exprs and "ipa_c_dim" in plan.aux_exprs
  )
  if has_ipa:
    lines.append(f"  const {ct} pk_ipa_gamma = ({ct})({em.expr(plan.aux_exprs['ipa_gamma'])});")
    lines.append(f"  const {idx_t} pk_C = ({idx_t})({em.expr(plan.aux_exprs['ipa_c_dim'])});")
  has_bias = "bias" in plan.stages
  if plan.mask_fill is not None:
    lines.append(f"  const {ct} pk_mfill = ({ct})({em.expr(plan.mask_fill)});")

  dq_buf = plan.out_bufs["dq"]
  dk_buf = plan.out_bufs["dk"]
  dv_buf = plan.out_bufs["dv"]
  lse_buf = plan.aux_exprs.get("lse_buf")
  lse_off = plan.aux_exprs.get("lse_offset")
  m_buf = plan.aux_exprs.get("m_buf")
  m_off = plan.aux_exprs.get("m_offset")
  l_buf = plan.aux_exprs.get("l_buf")
  l_off = plan.aux_exprs.get("l_offset")
  d_buf = plan.aux_exprs["d_buf"]
  d_off = plan.aux_exprs["d_offset"]
  assert isinstance(d_buf, Var) and (isinstance(lse_buf, Var) or (isinstance(m_buf, Var) and isinstance(l_buf, Var)))
  act_q = int(plan.aux_exprs["active_q"].value) if isinstance(plan.aux_exprs.get("active_q"), ir.Const) else 1
  act_k = int(plan.aux_exprs["active_k"].value) if isinstance(plan.aux_exprs.get("active_k"), ir.Const) else 1
  act_v = int(plan.aux_exprs["active_v"].value) if isinstance(plan.aux_exprs.get("active_v"), ir.Const) else 1
  act_b = int(plan.aux_exprs["active_b"].value) if isinstance(plan.aux_exprs.get("active_b"), ir.Const) else 0
  act_xq = int(plan.aux_exprs["active_xq"].value) if isinstance(plan.aux_exprs.get("active_xq"), ir.Const) else 0
  act_xk = int(plan.aux_exprs["active_xk"].value) if isinstance(plan.aux_exprs.get("active_xk"), ir.Const) else 0
  db_buf = plan.out_bufs.get("db")
  dxq_buf = plan.out_bufs.get("dxq")
  dxk_buf = plan.out_bufs.get("dxk")
  if has_bias:
    bs_expr = plan.aux_exprs.get("bias_sign", ir.Const(1.0, elt))
    lines.append(f"  const {ct} pk_bsign = ({ct})({em.expr(bs_expr)});")

  s_pad = MONOID_BM + 1
  k_pad = MONOID_BK + 1
  kv_fast_pad = 65
  lines.append(f"  {sh_qual} {ct} pk_sh_a[{MONOID_BM * kv_fast_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_b[{MONOID_BN * kv_fast_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_v[{MONOID_BN * kv_fast_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_p[{MONOID_BN * s_pad}];")  # P[j, i], reused for dS[j, i] in step 5
  lines.append(f"  {sh_qual} {ct} pk_sh_lse[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_d[{MONOID_BM}];")
  has_mask_buf = plan.mask_buf is not None and plan.mask_index is not None and plan.mask_fill is not None
  has_mask_cond = plan.mask_cond is not None and plan.mask_fill is not None
  if has_mask_buf:
    lines.append(f"  {sh_qual} unsigned char pk_sh_mask[{MONOID_BN * MONOID_BM}];")
    lines.append(f"  {sh_qual} int pk_any_mask;")

  # Hoist K_tile[32 x K] into pk_sh_b and V_tile[32 x D] into pk_sh_v ONCE when K,D <= 64!
  lines.append("  const bool pk_kv_hoisted = (pk_K <= 64 && pk_D <= 64);")
  lines.append("  if (pk_kv_hoisted) {")
  _emit_stage_load_2d(
    em,
    plan.stages["k"],
    "pk_sh_b",
    MONOID_BN,
    64,
    kv_fast_pad,
    "pk_j0",
    "0",
    "pk_N",
    "pk_K",
    ct,
    idx_t,
    lines,
    "    ",
    zero_lit,
  )
  _emit_stage_load_2d(
    em,
    plan.stages["v"],
    "pk_sh_v",
    MONOID_BN,
    64,
    kv_fast_pad,
    "pk_j0",
    "0",
    "pk_N",
    "pk_D",
    ct,
    idx_t,
    lines,
    "    ",
    zero_lit,
  )
  lines.append("  }")

  # Register accumulators for dK[32 x K] and dV[32 x D] (first 128 cols in registers -> zero atomics!)
  lines.append(f"  {ct} pk_dk_reg[{MAX_REG_D // 4}];")
  lines.append(f"  {ct} pk_dv_reg[{MAX_REG_D // 4}];")
  lines.append(
    f"  for (int pk_u = 0; pk_u < {MAX_REG_D // 4}; ++pk_u) {{ pk_dk_reg[pk_u] = {zero_lit}; pk_dv_reg[pk_u] = {zero_lit}; }}"
  )
  lines.append(f"  {barrier}")

  is_neg_inf_fill = isinstance(plan.mask_fill, ir.Const) and float(plan.mask_fill.value) == float("-inf")
  skip_eps = _lit(105.0 if elt.bits <= 32 else 750.0, elt, is_cuda)

  # Inner loop over Q tiles i0 = 0, 32, ... < M
  lines.append(f"  for ({idx_t} pk_i0 = 0; pk_i0 < pk_M; pk_i0 += {MONOID_BM}) {{")
  if plan.tile_false_guard is not None and plan.tile_origin_vars is not None and is_neg_inf_fill:
    i0_v, j0_v = plan.tile_origin_vars
    lines.append(f"    const {idx_t} {i0_v.name} = pk_i0;")
    lines.append(f"    const {idx_t} {j0_v.name} = pk_j0;")
    lines.append(f"    if ({em.expr(plan.tile_false_guard)}) continue;")

  # Load lse_tile[32] and D_tile[32]
  log_fn = "logf" if is_cuda and elt.name == "float32" else "log"
  lines.append(f"    if (pk_tid < {MONOID_BM}) {{")
  lines.append("      const " + idx_t + " pk_qi = pk_i0 + pk_tid;")
  lines.append("      if (pk_qi < pk_M) {")
  lines.append(f"        const {idx_t} pk_qflat = pk_batch_id * pk_M + pk_qi;")
  if isinstance(lse_buf, Var) and lse_off is not None:
    lines.append(f"        pk_sh_lse[pk_tid] = {em.data(lse_buf)}[({idx_t})({em.expr(lse_off)}) + pk_qflat];")
  else:
    assert isinstance(m_buf, Var) and m_off is not None and isinstance(l_buf, Var) and l_off is not None
    lines.append(
      f"        pk_sh_lse[pk_tid] = {em.data(m_buf)}[({idx_t})({em.expr(m_off)}) + pk_qflat] + "
      f"{log_fn}({em.data(l_buf)}[({idx_t})({em.expr(l_off)}) + pk_qflat]);"
    )
  lines.append(f"        pk_sh_d[pk_tid] = {em.data(d_buf)}[({idx_t})({em.expr(d_off)}) + pk_qflat];")
  lines.append("      } else {")
  lines.append(f"        pk_sh_lse[pk_tid] = {zero_lit}; pk_sh_d[pk_tid] = {zero_lit};")
  lines.append("      }")
  lines.append("    }")

  if has_mask_buf:
    assert plan.mask_index_vars is not None and plan.mask_buf is not None and plan.mask_index is not None
    mi_v, mj_v = plan.mask_index_vars
    lines.append("    if (pk_tid == 0) pk_any_mask = 0;")
    lines.append(f"    {barrier}")
    lines.append(f"    for ({idx_t} pk_ml = pk_tid; pk_ml < {MONOID_BN * MONOID_BM}; pk_ml += {MONOID_THREADS}) {{")
    lines.append("      const " + idx_t + " pk_mj = pk_ml >> 5;")
    lines.append("      const " + idx_t + " pk_mi = pk_ml & 31;")
    lines.append(f"      const {idx_t} {mi_v.name} = pk_i0 + pk_mi;")
    lines.append(f"      const {idx_t} {mj_v.name} = pk_j0 + pk_mj;")
    lines.append("      unsigned char pk_mv = 0;")
    lines.append(f"      if ({mi_v.name} < pk_M && {mj_v.name} < pk_N) {{")
    lines.append(f"        pk_mv = {em.data(plan.mask_buf)}[{em.expr(plan.mask_index)}] ? 1 : 0;")
    lines.append("        if (pk_mv) pk_any_mask = 1;")
    lines.append("      }")
    lines.append("      pk_sh_mask[pk_ml] = pk_mv;")
    lines.append("    }")
    lines.append(f"    {barrier}")
    # Exact skip when mask is all-false AND every row's lse >> mask_fill (exp(fill - lse) == 0!)
    lines.append("    if (pk_any_mask == 0) {")
    if is_neg_inf_fill:
      lines.append("      continue;")
    else:
      lines.append("      int pk_can_skip = 1;")
      lines.append(f"      for ({idx_t} pk_qi = 0; pk_qi < {MONOID_BM}; ++pk_qi) {{")
      lines.append(
        f"        if (pk_i0 + pk_qi < pk_M && !(pk_sh_lse[pk_qi] > pk_mfill + {skip_eps})) {{ pk_can_skip = 0; break; }}"
      )
      lines.append("      }")
      lines.append("      if (pk_can_skip) continue;")
    lines.append("    }")
  else:
    lines.append(f"    {barrier}")
    if plan.tile_false_guard is not None and plan.tile_origin_vars is not None and not is_neg_inf_fill:
      i0_v, j0_v = plan.tile_origin_vars
      lines.append(f"    const {idx_t} {i0_v.name} = pk_i0;")
      lines.append(f"    const {idx_t} {j0_v.name} = pk_j0;")
      lines.append(f"    if ({em.expr(plan.tile_false_guard)}) {{")
      lines.append("      int pk_can_skip = 1;")
      lines.append(f"      for ({idx_t} pk_qi = 0; pk_qi < {MONOID_BM}; ++pk_qi) {{")
      lines.append(
        f"        if (pk_i0 + pk_qi < pk_M && !(pk_sh_lse[pk_qi] > pk_mfill + {skip_eps})) {{ pk_can_skip = 0; break; }}"
      )
      lines.append("      }")
      lines.append("      if (pk_can_skip) continue;")
      lines.append("    }")

  # 1. Recompute S[j, i] = Q[i] . K[j] and dP[j, i] = dO[i] . V[j]
  # Fast hoisted-KV path (K, D <= 64): K & V never reloaded, dO loaded once and reused for dV!
  lines.append(f"    {ct} pk_s_acc[8];")
  lines.append(f"    {ct} pk_dp_acc[8];")
  lines.append(
    f"    for (int pk_c = 0; pk_c < 8; ++pk_c) {{ pk_s_acc[pk_c] = {zero_lit}; pk_dp_acc[pk_c] = {zero_lit}; }}"
  )
  lines.append("    if (pk_kv_hoisted) {")
  _emit_stage_load_2d(
    em,
    plan.stages["q"],
    "pk_sh_a",
    MONOID_BM,
    64,
    kv_fast_pad,
    "pk_i0",
    "0",
    "pk_M",
    "pk_K",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  lines.append(f"      {barrier}")
  lines.append("      #pragma unroll")
  lines.append("      for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("        const " + idx_t + " pk_ic = pk_c * 4 + pk_t;")
  lines.append(f"        {ct} pk_dot = {zero_lit};")
  lines.append("        for (" + idx_t + " pk_kk = 0; pk_kk < pk_K; ++pk_kk) {")
  lines.append(f"          pk_dot += pk_sh_a[pk_ic * {kv_fast_pad} + pk_kk] * pk_sh_b[pk_r * {kv_fast_pad} + pk_kk];")
  lines.append("        }")
  lines.append("        pk_s_acc[pk_c] = pk_dot;")
  lines.append("      }")
  lines.append(f"      {barrier}")
  _emit_stage_load_2d(
    em,
    plan.stages["do"],
    "pk_sh_a",
    MONOID_BM,
    64,
    kv_fast_pad,
    "pk_i0",
    "0",
    "pk_M",
    "pk_D",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  lines.append(f"      {barrier}")
  lines.append("      #pragma unroll")
  lines.append("      for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("        const " + idx_t + " pk_ic = pk_c * 4 + pk_t;")
  lines.append(f"        {ct} pk_dot = {zero_lit};")
  lines.append("        for (" + idx_t + " pk_dd = 0; pk_dd < pk_D; ++pk_dd) {")
  lines.append(f"          pk_dot += pk_sh_a[pk_ic * {kv_fast_pad} + pk_dd] * pk_sh_v[pk_r * {kv_fast_pad} + pk_dd];")
  lines.append("        }")
  lines.append("        pk_dp_acc[pk_c] = pk_dot;")
  lines.append("      }")
  lines.append("    } else {")
  lines.append(f"      for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {MONOID_BK}) {{")
  _emit_stage_load_2d(
    em,
    plan.stages["q"],
    "pk_sh_a",
    MONOID_BM,
    MONOID_BK,
    k_pad,
    "pk_i0",
    "pk_k0",
    "pk_M",
    "pk_K",
    ct,
    idx_t,
    lines,
    "        ",
    zero_lit,
  )
  _emit_stage_load_2d(
    em,
    plan.stages["k"],
    "pk_sh_b",
    MONOID_BN,
    MONOID_BK,
    k_pad,
    "pk_j0",
    "pk_k0",
    "pk_N",
    "pk_K",
    ct,
    idx_t,
    lines,
    "        ",
    zero_lit,
  )
  lines.append(f"        {barrier}")
  lines.append(f"        const {idx_t} pk_klen = pk_K - pk_k0 < {MONOID_BK} ? pk_K - pk_k0 : {MONOID_BK};")
  lines.append("        for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("          const " + idx_t + " pk_ic = pk_c * 4 + pk_t;")
  lines.append(f"          {ct} pk_dot = pk_s_acc[pk_c];")
  lines.append(f"          for ({idx_t} pk_kk = 0; pk_kk < pk_klen; ++pk_kk) {{")
  lines.append(f"            pk_dot += pk_sh_a[pk_ic * {k_pad} + pk_kk] * pk_sh_b[pk_r * {k_pad} + pk_kk];")
  lines.append("          }")
  lines.append("          pk_s_acc[pk_c] = pk_dot;")
  lines.append("        }")
  lines.append(f"        {barrier}")
  lines.append("      }")
  lines.append(f"      for ({idx_t} pk_d0 = 0; pk_d0 < pk_D; pk_d0 += {MONOID_BK}) {{")
  _emit_stage_load_2d(
    em,
    plan.stages["do"],
    "pk_sh_a",
    MONOID_BM,
    MONOID_BK,
    k_pad,
    "pk_i0",
    "pk_d0",
    "pk_M",
    "pk_D",
    ct,
    idx_t,
    lines,
    "        ",
    zero_lit,
  )
  _emit_stage_load_2d(
    em,
    plan.stages["v"],
    "pk_sh_b",
    MONOID_BN,
    MONOID_BK,
    k_pad,
    "pk_j0",
    "pk_d0",
    "pk_N",
    "pk_D",
    ct,
    idx_t,
    lines,
    "        ",
    zero_lit,
  )
  lines.append(f"        {barrier}")
  lines.append(f"        const {idx_t} pk_dlen = pk_D - pk_d0 < {MONOID_BK} ? pk_D - pk_d0 : {MONOID_BK};")
  lines.append("        for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("          const " + idx_t + " pk_ic = pk_c * 4 + pk_t;")
  lines.append(f"          {ct} pk_dot = pk_dp_acc[pk_c];")
  lines.append(f"          for ({idx_t} pk_dd = 0; pk_dd < pk_dlen; ++pk_dd) {{")
  lines.append(f"            pk_dot += pk_sh_a[pk_ic * {k_pad} + pk_dd] * pk_sh_b[pk_r * {k_pad} + pk_dd];")
  lines.append("          }")
  lines.append("          pk_dp_acc[pk_c] = pk_dot;")
  lines.append("        }")
  lines.append(f"        {barrier}")
  lines.append("      }")
  lines.append("    }")

  # 3. Compute P[j, i] (in pk_sh_p) and dS[j, i] (in registers pk_ds_reg[8], staged to pk_sh_p after dV)
  lines.append(f"    {ct} pk_ds_reg[8];")
  lines.append("    for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("      const " + idx_t + " pk_ic = pk_c * 4 + pk_t;")
  lines.append("      const " + idx_t + " pk_gi = pk_i0 + pk_ic;")
  lines.append(f"      {ct} pk_pv = {zero_lit};")
  lines.append(f"      {ct} pk_dsv = {zero_lit};")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_N) {")
  lines.append("        " + ct + " pk_sv = pk_s_acc[pk_c];")
  if plan.score_scale is not None:
    lines.append("        pk_sv *= pk_scale;")
  if has_ipa:
    lines.append(f"        {ct} pk_sq = {zero_lit};")
    lines.append(f"        for ({idx_t} pk_cc = 0; pk_cc < pk_C; ++pk_cc) {{")
    _emit_inline_stage_2d(em, plan.stages["xq"], "pk_gi", "pk_cc", "pk_xqv", ct, idx_t, lines, "          ")
    _emit_inline_stage_2d(em, plan.stages["xk"], "pk_gj", "pk_cc", "pk_xkv", ct, idx_t, lines, "          ")
    lines.append(f"          const {ct} pk_df = pk_xqv - pk_xkv;")
    lines.append("          pk_sq += pk_df * pk_df;")
    lines.append("        }")
    lines.append("        pk_sv -= pk_ipa_gamma * pk_sq;")
  if has_bias:
    _emit_inline_stage_2d(em, plan.stages["bias"], "pk_gi", "pk_gj", "pk_bv", ct, idx_t, lines, "        ")
    lines.append("        pk_sv += pk_bsign * pk_bv;")
  lines.append("        bool pk_keep = true;")
  if has_mask_buf:
    lines.append("        pk_keep = (pk_sh_mask[pk_r * 32 + pk_ic] != 0);")
    lines.append("        if (!pk_keep) pk_sv = pk_mfill;")
  elif has_mask_cond:
    assert plan.mask_index_vars is not None and plan.mask_cond is not None
    mi_v, mj_v = plan.mask_index_vars
    lines.append(f"        const {idx_t} {mi_v.name} = pk_gi;")
    lines.append(f"        const {idx_t} {mj_v.name} = pk_gj;")
    lines.append(f"        pk_keep = ({em.expr(plan.mask_cond)});")
    lines.append("        if (!pk_keep) pk_sv = pk_mfill;")
  lines.append(f"        pk_pv = {expf}(pk_sv - pk_sh_lse[pk_ic]);")
  lines.append("        if (pk_keep) {")
  lines.append(f"          const {ct} pk_dsu = pk_pv * (pk_dp_acc[pk_c] - pk_sh_d[pk_ic]);")
  if act_b and db_buf is not None:
    lines.append(f"          {em.data(db_buf)}[(pk_batch_id * pk_M + pk_gi) * pk_N + pk_gj] = pk_bsign * pk_dsu;")
  if has_ipa and ((act_xq and dxq_buf is not None) or (act_xk and dxk_buf is not None)):
    lines.append(f"          const {ct} pk_dc_coef = ({_lit(-2.0, elt, is_cuda)} * pk_ipa_gamma) * pk_dsu;")
    lines.append(f"          for ({idx_t} pk_cc = 0; pk_cc < pk_C; ++pk_cc) {{")
    _emit_inline_stage_2d(em, plan.stages["xq"], "pk_gi", "pk_cc", "pk_xqv2", ct, idx_t, lines, "            ")
    _emit_inline_stage_2d(em, plan.stages["xk"], "pk_gj", "pk_cc", "pk_xkv2", ct, idx_t, lines, "            ")
    lines.append(f"            const {ct} pk_gc = pk_dc_coef * (pk_xqv2 - pk_xkv2);")
    if act_xq and dxq_buf is not None:
      _emit_atomic_add(
        is_cuda,
        f"&{em.data(dxq_buf)}[(pk_batch_id * pk_M + pk_gi) * pk_C + pk_cc]",
        "pk_gc",
        elt,
        lines,
        "            ",
      )
    if act_xk and dxk_buf is not None:
      _emit_atomic_add(
        is_cuda,
        f"&{em.data(dxk_buf)}[(pk_batch_id * pk_N + pk_gj) * pk_C + pk_cc]",
        "-pk_gc",
        elt,
        lines,
        "            ",
      )
    lines.append("          }")
  lines.append("          pk_dsv = pk_dsu;")
  if plan.score_scale is not None:
    lines.append("          pk_dsv *= pk_scale;")
  lines.append("        }")
  lines.append("      }")
  lines.append(f"      pk_sh_p[pk_r * {s_pad} + pk_ic] = pk_pv;")
  lines.append("      pk_ds_reg[pk_c] = pk_dsv;")
  lines.append("    }")
  lines.append(f"    {barrier}")
  lines.append(f"    const {idx_t} pk_ilen = pk_M - pk_i0 < {MONOID_BM} ? pk_M - pk_i0 : {MONOID_BM};")
  lines.append(f"    const {idx_t} pk_jlen = pk_N - pk_j0 < {MONOID_BN} ? pk_N - pk_j0 : {MONOID_BN};")

  # 4. Accumulate dV[j, d] += sum_i P[j, i] * dO[i, d] (reuses pk_sh_a = dO when pk_kv_hoisted!)
  if act_v:
    lines.append("    if (pk_kv_hoisted) {")
    lines.append("      if (pk_gj < pk_N) {")
    lines.append("        for (int pk_u = 0; pk_u < 16; ++pk_u) {")
    lines.append("          const " + idx_t + " pk_gd = pk_u * 4 + pk_t;")
    lines.append("          if (pk_gd < pk_D) {")
    lines.append(f"            {ct} pk_acc = {zero_lit};")
    lines.append(f"            for ({idx_t} pk_ii = 0; pk_ii < pk_ilen; ++pk_ii) {{")
    lines.append(f"              pk_acc += pk_sh_p[pk_r * {s_pad} + pk_ii] * pk_sh_a[pk_ii * {kv_fast_pad} + pk_gd];")
    lines.append("            }")
    lines.append("            pk_dv_reg[pk_u] += pk_acc;")
    lines.append("          }")
    lines.append("        }")
    lines.append("      }")
    lines.append(f"      {barrier}")
    lines.append("    } else {")
    lines.append(f"      for ({idx_t} pk_d0 = 0; pk_d0 < pk_D; pk_d0 += {MONOID_BK}) {{")
    _emit_stage_load_2d(
      em,
      plan.stages["do"],
      "pk_sh_a",
      MONOID_BM,
      MONOID_BK,
      k_pad,
      "pk_i0",
      "pk_d0",
      "pk_M",
      "pk_D",
      ct,
      idx_t,
      lines,
      "        ",
      zero_lit,
    )
    lines.append(f"        {barrier}")
    lines.append("        for (int pk_c = 0; pk_c < 8; ++pk_c) {")
    lines.append("          const " + idx_t + " pk_dc = pk_c * 4 + pk_t;")
    lines.append("          const " + idx_t + " pk_gd = pk_d0 + pk_dc;")
    lines.append("          if (pk_gj < pk_N && pk_gd < pk_D) {")
    lines.append(f"            {ct} pk_acc = {zero_lit};")
    lines.append(f"            for ({idx_t} pk_ii = 0; pk_ii < pk_ilen; ++pk_ii) {{")
    lines.append(f"              pk_acc += pk_sh_p[pk_r * {s_pad} + pk_ii] * pk_sh_a[pk_ii * {k_pad} + pk_dc];")
    lines.append("            }")
    lines.append(f"            if (pk_d0 < {MAX_REG_D}) {{")
    lines.append("              pk_dv_reg[(int)((pk_d0 >> 2) + pk_c)] += pk_acc;")
    lines.append("            } else {")
    lines.append(f"              {em.data(dv_buf)}[pk_jrow_flat * pk_D + pk_gd] += pk_acc;")
    lines.append("            }")
    lines.append("          }")
    lines.append("        }")
    lines.append(f"        {barrier}")
    lines.append("      }")
    lines.append("    }")

  # 5. Accumulate dK[j, k] += sum_i dS[j, i] * Q[i, k]  AND  dQ[i, k] += atomicAdd(sum_j dS[j, i] * K[j, k])
  # Stage dS[j, i] from registers into pk_sh_p (P is no longer needed!)
  if act_k or act_q:
    lines.append("    for (int pk_c = 0; pk_c < 8; ++pk_c) {")
    lines.append(f"      pk_sh_p[pk_r * {s_pad} + (pk_c * 4 + pk_t)] = pk_ds_reg[pk_c];")
    lines.append("    }")
    lines.append(f"    {barrier}")
    lines.append("    if (pk_kv_hoisted) {")
    if act_k:
      _emit_stage_load_2d(
        em,
        plan.stages["q"],
        "pk_sh_a",
        MONOID_BM,
        64,
        kv_fast_pad,
        "pk_i0",
        "0",
        "pk_M",
        "pk_K",
        ct,
        idx_t,
        lines,
        "      ",
        zero_lit,
      )
      lines.append(f"      {barrier}")
      lines.append("      if (pk_gj < pk_N) {")
      lines.append("        for (int pk_u = 0; pk_u < 16; ++pk_u) {")
      lines.append("          const " + idx_t + " pk_gk = pk_u * 4 + pk_t;")
      lines.append("          if (pk_gk < pk_K) {")
      lines.append(f"            {ct} pk_acc = {zero_lit};")
      lines.append(f"            for ({idx_t} pk_ii = 0; pk_ii < pk_ilen; ++pk_ii) {{")
      lines.append(f"              pk_acc += pk_sh_p[pk_r * {s_pad} + pk_ii] * pk_sh_a[pk_ii * {kv_fast_pad} + pk_gk];")
      lines.append("            }")
      lines.append("            pk_dk_reg[pk_u] += pk_acc;")
      lines.append("          }")
      lines.append("        }")
      lines.append("      }")
    if act_q:
      lines.append("      {")
      lines.append("        const " + idx_t + " pk_qi = pk_i0 + pk_r;")
      lines.append("        if (pk_qi < pk_M) {")
      lines.append("          const " + idx_t + " pk_qrow_flat = pk_batch_id * pk_M + pk_qi;")
      lines.append("          for (int pk_u = 0; pk_u < 16; ++pk_u) {")
      lines.append("            const " + idx_t + " pk_gk = pk_u * 4 + pk_t;")
      lines.append("            if (pk_gk < pk_K) {")
      lines.append(f"              {ct} pk_dqv = {zero_lit};")
      lines.append(f"              for ({idx_t} pk_jj = 0; pk_jj < pk_jlen; ++pk_jj) {{")
      lines.append(
        f"                pk_dqv += pk_sh_p[pk_jj * {s_pad} + pk_r] * pk_sh_b[pk_jj * {kv_fast_pad} + pk_gk];"
      )
      lines.append("              }")
      _emit_atomic_add(
        is_cuda, f"&{em.data(dq_buf)}[pk_qrow_flat * pk_K + pk_gk]", "pk_dqv", elt, lines, "              "
      )
      lines.append("            }")
      lines.append("          }")
      lines.append("        }")
      lines.append("      }")
    lines.append(f"      {barrier}")
    lines.append("    } else {")
    lines.append(f"      for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {MONOID_BK}) {{")
    if act_k:
      _emit_stage_load_2d(
        em,
        plan.stages["q"],
        "pk_sh_a",
        MONOID_BM,
        MONOID_BK,
        k_pad,
        "pk_i0",
        "pk_k0",
        "pk_M",
        "pk_K",
        ct,
        idx_t,
        lines,
        "        ",
        zero_lit,
      )
    if act_q:
      _emit_stage_load_2d(
        em,
        plan.stages["k"],
        "pk_sh_b",
        MONOID_BN,
        MONOID_BK,
        k_pad,
        "pk_j0",
        "pk_k0",
        "pk_N",
        "pk_K",
        ct,
        idx_t,
        lines,
        "        ",
        zero_lit,
      )
    lines.append(f"        {barrier}")
    if act_k:
      lines.append("        for (int pk_c = 0; pk_c < 8; ++pk_c) {")
      lines.append("          const " + idx_t + " pk_kc = pk_c * 4 + pk_t;")
      lines.append("          const " + idx_t + " pk_gk = pk_k0 + pk_kc;")
      lines.append("          if (pk_gj < pk_N && pk_gk < pk_K) {")
      lines.append(f"            {ct} pk_acc = {zero_lit};")
      lines.append(f"            for ({idx_t} pk_ii = 0; pk_ii < pk_ilen; ++pk_ii) {{")
      lines.append(f"              pk_acc += pk_sh_p[pk_r * {s_pad} + pk_ii] * pk_sh_a[pk_ii * {k_pad} + pk_kc];")
      lines.append("            }")
      lines.append(f"            if (pk_k0 < {MAX_REG_D}) {{")
      lines.append("              pk_dk_reg[(int)((pk_k0 >> 2) + pk_c)] += pk_acc;")
      lines.append("            } else {")
      lines.append(f"              {em.data(dk_buf)}[pk_jrow_flat * pk_K + pk_gk] += pk_acc;")
      lines.append("            }")
      lines.append("          }")
      lines.append("        }")
    if act_q:
      lines.append("        {")
      lines.append("          const " + idx_t + " pk_qi = pk_i0 + pk_r;")
      lines.append("          if (pk_qi < pk_M) {")
      lines.append("            const " + idx_t + " pk_qrow_flat = pk_batch_id * pk_M + pk_qi;")
      lines.append("            for (int pk_c = 0; pk_c < 8; ++pk_c) {")
      lines.append("              const " + idx_t + " pk_kc = pk_c * 4 + pk_t;")
      lines.append("              const " + idx_t + " pk_gk = pk_k0 + pk_kc;")
      lines.append("              if (pk_gk < pk_K) {")
      lines.append(f"                {ct} pk_dqv = {zero_lit};")
      lines.append(f"                for ({idx_t} pk_jj = 0; pk_jj < pk_jlen; ++pk_jj) {{")
      lines.append(f"                  pk_dqv += pk_sh_p[pk_jj * {s_pad} + pk_r] * pk_sh_b[pk_jj * {k_pad} + pk_kc];")
      lines.append("                }")
      _emit_atomic_add(
        is_cuda, f"&{em.data(dq_buf)}[pk_qrow_flat * pk_K + pk_gk]", "pk_dqv", elt, lines, "                "
      )
      lines.append("              }")
      lines.append("            }")
      lines.append("          }")
      lines.append("        }")
    lines.append(f"        {barrier}")
    lines.append("      }")
    lines.append("    }")

  lines.append("  }")  # end i0 loop

  # Store register-accumulated dK and dV once to HBM!
  lines.append("  if (pk_gj < pk_N) {")
  if act_k:
    lines.append(f"    for (int pk_u = 0; pk_u < {MAX_REG_D // 4}; ++pk_u) {{")
    lines.append("      const " + idx_t + " pk_gk = pk_u * 4 + pk_t;")
    lines.append("      if (pk_gk < pk_K) {")
    lines.append(f"        {em.data(dk_buf)}[pk_jrow_flat * pk_K + pk_gk] = pk_dk_reg[pk_u];")
    lines.append("      }")
    lines.append("    }")
  if act_v:
    lines.append(f"    for (int pk_u = 0; pk_u < {MAX_REG_D // 4}; ++pk_u) {{")
    lines.append("      const " + idx_t + " pk_gd = pk_u * 4 + pk_t;")
    lines.append("      if (pk_gd < pk_D) {")
    lines.append(f"        {em.data(dv_buf)}[pk_jrow_flat * pk_D + pk_gd] = pk_dv_reg[pk_u];")
    lines.append("      }")
    lines.append("    }")
  lines.append("  }")
  lines.append("}")

  written = {b.uid for b in plan.out_bufs.values()}
  grid_shape = [*plan.batch_shape, plan.n_dim, ir.Const(1, ops.I)]
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in grid_shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={
      "tiled": {
        "bm": MONOID_BN,
        "bn": 1,
        "bk": MONOID_BK,
        "tm": 1,
        "tn": 1,
        "threads": MONOID_THREADS,
        "shape": shape_meta,
      }
    },
  )


def _generate_monoid_lse(
  loop: ParLoop,
  plan: MonoidalTilePlan,
  free: list[Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None,
) -> KernelPlan:
  """Generate 1-pass Cut Cross-Entropy / pairwise LogSumExp forward GPU kernel."""
  is_cuda = target == "cuda"
  elt = plan.elt
  ct = em.ctype(elt)
  zero_lit = _lit(0.0, elt, is_cuda)
  neg_inf = _lit(float("-inf"), elt, is_cuda)
  expf = _exp_fn(elt, is_cuda)
  logf = _log_fn(elt, is_cuda)
  fmax = _max_fn(elt, is_cuda)

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  lines, idx_t, sh_qual, barrier, _ = _emit_header(em, target, prelude, scalars, bufs)
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.m_dim)});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.n_dim)});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.k_dim)});")
  _emit_batch_unravel(em, plan, "pk_M", MONOID_BM, idx_t, lines)
  lines.append(f"  const {idx_t} pk_i0 = pk_blk_r * {MONOID_BM};")
  lines.append(f"  const {idx_t} pk_gi = pk_i0 + pk_r;")
  lines.append(f"  const {idx_t} pk_row_flat = pk_batch_id * pk_M + pk_gi;")
  if plan.score_scale is not None:
    lines.append(f"  const {ct} pk_scale = ({ct})({em.expr(plan.score_scale)});")
  has_ipa = (
    "xq" in plan.stages and "xk" in plan.stages and "ipa_gamma" in plan.aux_exprs and "ipa_c_dim" in plan.aux_exprs
  )
  if has_ipa:
    lines.append(f"  const {ct} pk_ipa_gamma = ({ct})({em.expr(plan.aux_exprs['ipa_gamma'])});")
    lines.append(f"  const {idx_t} pk_C = ({idx_t})({em.expr(plan.aux_exprs['ipa_c_dim'])});")
  has_bias = "bias" in plan.stages

  k_pad = MONOID_BK + 1
  s_pad = MONOID_BN + 1
  lines.append(f"  {sh_qual} {ct} pk_sh_a[{MONOID_BM * k_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_b[{MONOID_BN * k_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_s[{MONOID_BM * s_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_m[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_l[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_red[{MONOID_BM * 4}];")

  has_target = "target_buf" in plan.aux_exprs and "target_index" in plan.aux_exprs
  if has_target:
    tbuf = plan.aux_exprs["target_buf"]
    tidx = plan.aux_exprs["target_index"]
    assert isinstance(tbuf, Var)
    lines.append(f"  {sh_qual} {idx_t} pk_sh_tgt[{MONOID_BM}];")
    lines.append(f"  {sh_qual} {ct} pk_sh_pick[{MONOID_BM}];")
    lines.append("  if (pk_t == 0) {")
    lines.append(f"    pk_sh_pick[pk_r] = {zero_lit};")
    lines.append("    if (pk_gi < pk_M) {")
    # Bind origin var if tidx references it
    t_orig = next((x for x in ir.walk_expr(tidx) if isinstance(x, Var) and x.name.startswith("lo_")), None)
    if t_orig is not None:
      lines.append(f"      const {idx_t} {t_orig.name} = pk_gi;")
    lines.append(f"      {idx_t} pk_tv = ({idx_t}){em.data(tbuf)}[{em.expr(tidx)}];")
    lines.append("      if (pk_tv < 0) pk_tv += pk_N;")
    lines.append("      pk_sh_tgt[pk_r] = pk_tv;")
    lines.append("    } else {")
    lines.append("      pk_sh_tgt[pk_r] = -1;")
    lines.append("    }")
    lines.append("  }")

  lines.append("  if (pk_t == 0) {")
  lines.append(f"    pk_sh_m[pk_r] = {neg_inf};")
  lines.append(f"    pk_sh_l[pk_r] = {zero_lit};")
  lines.append("  }")
  lines.append(f"  {barrier}")

  lines.append(f"  for ({idx_t} pk_j0 = 0; pk_j0 < pk_N; pk_j0 += {MONOID_BN}) {{")
  lines.append(f"    {ct} pk_s_acc[8];")
  lines.append(f"    for (int pk_c = 0; pk_c < 8; ++pk_c) pk_s_acc[pk_c] = {zero_lit};")
  lines.append(f"    for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {MONOID_BK}) {{")
  _emit_stage_load_2d(
    em,
    plan.stages["h"],
    "pk_sh_a",
    MONOID_BM,
    MONOID_BK,
    k_pad,
    "pk_i0",
    "pk_k0",
    "pk_M",
    "pk_K",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  _emit_stage_load_2d(
    em,
    plan.stages["w"],
    "pk_sh_b",
    MONOID_BN,
    MONOID_BK,
    k_pad,
    "pk_j0",
    "pk_k0",
    "pk_N",
    "pk_K",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  lines.append(f"      {barrier}")
  lines.append(f"      const {idx_t} pk_klen = pk_K - pk_k0 < {MONOID_BK} ? pk_K - pk_k0 : {MONOID_BK};")
  lines.append("      for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("        const " + idx_t + " pk_jc = pk_c * 4 + pk_t;")
  lines.append(f"        {ct} pk_dot = pk_s_acc[pk_c];")
  lines.append(f"        for ({idx_t} pk_kk = 0; pk_kk < pk_klen; ++pk_kk) {{")
  if plan.score_op == "sqdist":
    lines.append(f"          {ct} pk_df = pk_sh_a[pk_r * {k_pad} + pk_kk] - pk_sh_b[pk_jc * {k_pad} + pk_kk];")
    lines.append("          pk_dot += pk_df * pk_df;")
  else:
    lines.append(f"          pk_dot += pk_sh_a[pk_r * {k_pad} + pk_kk] * pk_sh_b[pk_jc * {k_pad} + pk_kk];")
  lines.append("        }")
  lines.append("        pk_s_acc[pk_c] = pk_dot;")
  lines.append("      }")
  lines.append(f"      {barrier}")
  lines.append("    }")

  lines.append(f"    {ct} pk_lmax = {neg_inf};")
  lines.append("    for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("      const " + idx_t + " pk_jc = pk_c * 4 + pk_t;")
  lines.append("      const " + idx_t + " pk_gj = pk_j0 + pk_jc;")
  lines.append(f"      {ct} pk_sv = {neg_inf};")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_N) {")
  lines.append("        pk_sv = pk_s_acc[pk_c];")
  if plan.score_scale is not None:
    lines.append("        pk_sv *= pk_scale;")
  if has_ipa:
    lines.append(f"        {ct} pk_sq = {zero_lit};")
    lines.append(f"        for ({idx_t} pk_cc = 0; pk_cc < pk_C; ++pk_cc) {{")
    _emit_inline_stage_2d(em, plan.stages["xq"], "pk_gi", "pk_cc", "pk_xqv", ct, idx_t, lines, "          ")
    _emit_inline_stage_2d(em, plan.stages["xk"], "pk_gj", "pk_cc", "pk_xkv", ct, idx_t, lines, "          ")
    lines.append(f"          const {ct} pk_df = pk_xqv - pk_xkv;")
    lines.append("          pk_sq += pk_df * pk_df;")
    lines.append("        }")
    lines.append("        pk_sv -= pk_ipa_gamma * pk_sq;")
  if has_bias:
    _emit_inline_stage_2d(em, plan.stages["bias"], "pk_gi", "pk_gj", "pk_bv", ct, idx_t, lines, "        ")
    lines.append("        pk_sv += pk_bv;")
  if has_target:
    lines.append("        if (pk_gj == pk_sh_tgt[pk_r]) pk_sh_pick[pk_r] = pk_sv;")
  lines.append(f"        pk_lmax = {fmax}(pk_lmax, pk_sv);")
  lines.append("      }")
  lines.append(f"      pk_sh_s[pk_r * {s_pad} + pk_jc] = pk_sv;")
  lines.append("    }")
  lines.append("    pk_sh_red[pk_r * 4 + pk_t] = pk_lmax;")
  lines.append(f"    {barrier}")

  lines.append("    if (pk_t == 0 && pk_gi < pk_M) {")
  lines.append(
    f"      {ct} pk_tm = {fmax}({fmax}(pk_sh_red[pk_r*4], pk_sh_red[pk_r*4+1]), {fmax}(pk_sh_red[pk_r*4+2], pk_sh_red[pk_r*4+3]));"
  )
  lines.append("      if (pk_j0 == 0) {")
  lines.append("        pk_sh_m[pk_r] = pk_tm;")
  lines.append("      } else {")
  lines.append(f"        {ct} pk_mold = pk_sh_m[pk_r];")
  lines.append(f"        {ct} pk_mnew = {fmax}(pk_mold, pk_tm);")
  lines.append("        pk_sh_m[pk_r] = pk_mnew;")
  lines.append(f"        pk_sh_l[pk_r] *= {expf}(pk_mold - pk_mnew);")
  lines.append("      }")
  lines.append("    }")
  lines.append(f"    {barrier}")
  lines.append(f"    const {ct} pk_mcur = pk_sh_m[pk_r];")
  lines.append(f"    {ct} pk_lsum = {zero_lit};")
  lines.append("    for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("      const " + idx_t + " pk_jc = pk_c * 4 + pk_t;")
  lines.append("      const " + idx_t + " pk_gj = pk_j0 + pk_jc;")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_N) {")
  lines.append(f"        pk_lsum += {expf}(pk_sh_s[pk_r * {s_pad} + pk_jc] - pk_mcur);")
  lines.append("      }")
  lines.append("    }")
  lines.append("    pk_sh_red[pk_r * 4 + pk_t] = pk_lsum;")
  lines.append(f"    {barrier}")
  lines.append("    if (pk_t == 0 && pk_gi < pk_M) {")
  lines.append(
    "      pk_sh_l[pk_r] += (pk_sh_red[pk_r*4] + pk_sh_red[pk_r*4+1] + pk_sh_red[pk_r*4+2] + pk_sh_red[pk_r*4+3]);"
  )
  lines.append("    }")
  lines.append(f"    {barrier}")
  lines.append("  }")

  lines.append("  if (pk_t == 0 && pk_gi < pk_M) {")
  lines.append(f"    const {ct} pk_lse = pk_sh_m[pk_r] + {logf}(pk_sh_l[pk_r]);")
  if "m" in plan.out_bufs:
    lines.append(f"    {em.data(plan.out_bufs['m'])}[pk_row_flat] = pk_sh_m[pk_r];")
  if "l" in plan.out_bufs:
    lines.append(f"    {em.data(plan.out_bufs['l'])}[pk_row_flat] = pk_sh_l[pk_r];")
  if "out" in plan.out_bufs and has_target:
    lines.append(f"    {em.data(plan.out_bufs['out'])}[pk_row_flat] = pk_sh_pick[pk_r] - pk_lse;")
  if "picked_sub_m" in plan.out_bufs and has_target:
    lines.append(f"    {em.data(plan.out_bufs['picked_sub_m'])}[pk_row_flat] = pk_sh_pick[pk_r] - pk_sh_m[pk_r];")
  if "lse" in plan.out_bufs:
    lines.append(f"    {em.data(plan.out_bufs['lse'])}[pk_row_flat] = pk_lse;")
  lines.append("  }")
  lines.append("}")

  written = {b.uid for b in plan.out_bufs.values()}
  grid_shape = [*plan.batch_shape, plan.m_dim, ir.Const(1, ops.I)]
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in grid_shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={
      "tiled": {
        "bm": MONOID_BM,
        "bn": 1,
        "bk": MONOID_BK,
        "tm": 1,
        "tn": 1,
        "threads": MONOID_THREADS,
        "shape": shape_meta,
      }
    },
  )


def _generate_cce_bwd(
  loop: ParLoop,
  plan: MonoidalTilePlan,
  free: list[Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None,
) -> KernelPlan:
  """Generate SRAM-recomputed Cut Cross-Entropy backward GPU kernel (outer vocab tiles)."""
  is_cuda = target == "cuda"
  elt = plan.elt
  ct = em.ctype(elt)
  zero_lit = _lit(0.0, elt, is_cuda)
  one_lit = _lit(1.0, elt, is_cuda)
  expf = _exp_fn(elt, is_cuda)

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  # 2D grid over (ceil(M / 128), ceil(N / 32)): saturates SMs even when V is small (e.g. V=65)!
  cce_m_chunk = 128
  lines, idx_t, sh_qual, barrier, _ = _emit_header(em, target, prelude, scalars, bufs, need_atomic_f32=True)
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.m_dim)});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.n_dim)});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.k_dim)});")
  lines.append(f"  const {idx_t} pk_nb_n = (pk_N + {MONOID_BN - 1}) / {MONOID_BN};")
  lines.append("  const " + idx_t + " pk_blk_n = pk_blk % pk_nb_n;")
  lines.append("  const " + idx_t + " pk_blk_m = pk_blk / pk_nb_n;")
  lines.append(f"  const {idx_t} pk_j0 = pk_blk_n * {MONOID_BN};")
  lines.append(f"  const {idx_t} pk_m_start = pk_blk_m * {cce_m_chunk};")
  lines.append(f"  const {idx_t} pk_m_end = pk_m_start + {cce_m_chunk} < pk_M ? pk_m_start + {cce_m_chunk} : pk_M;")
  lines.append(f"  const {idx_t} pk_gv = pk_j0 + pk_r;")

  dh_buf = plan.out_bufs["dh"]
  dw_buf = plan.out_bufs["dw"]
  lse_buf = plan.aux_exprs.get("lse_buf")
  lse_off = plan.aux_exprs.get("lse_offset")
  m_buf = plan.aux_exprs.get("m_buf")
  m_off = plan.aux_exprs.get("m_offset")
  l_buf = plan.aux_exprs.get("l_buf")
  l_off = plan.aux_exprs.get("l_offset")
  g_buf = plan.aux_exprs["g_buf"]
  g_off = plan.aux_exprs["g_offset"]
  t_buf = plan.aux_exprs["target_buf"]
  t_off = plan.aux_exprs["target_offset"]
  assert (
    isinstance(g_buf, Var)
    and isinstance(t_buf, Var)
    and (isinstance(lse_buf, Var) or (isinstance(m_buf, Var) and isinstance(l_buf, Var)))
  )
  act_h = int(plan.aux_exprs["active_h"].value) if isinstance(plan.aux_exprs.get("active_h"), ir.Const) else 1
  act_w = int(plan.aux_exprs["active_w"].value) if isinstance(plan.aux_exprs.get("active_w"), ir.Const) else 1

  s_pad = MONOID_BM + 1
  k_pad = MONOID_BK + 1
  lines.append(f"  {sh_qual} {ct} pk_sh_a[{MONOID_BM * k_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_b[{MONOID_BN * k_pad}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_dl[{MONOID_BN * s_pad}];")  # dlog[v, i]
  lines.append(f"  {sh_qual} {ct} pk_sh_lse[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {ct} pk_sh_g[{MONOID_BM}];")
  lines.append(f"  {sh_qual} {idx_t} pk_sh_tgt[{MONOID_BM}];")

  lines.append(f"  {ct} pk_dw_reg[{MAX_REG_D // 4}];")
  lines.append(f"  for (int pk_u = 0; pk_u < {MAX_REG_D // 4}; ++pk_u) pk_dw_reg[pk_u] = {zero_lit};")

  log_fn = "logf" if is_cuda and elt.name == "float32" else "log"
  lines.append(f"  for ({idx_t} pk_i0 = pk_m_start; pk_i0 < pk_m_end; pk_i0 += {MONOID_BM}) {{")
  lines.append(f"    if (pk_tid < {MONOID_BM}) {{")
  lines.append("      const " + idx_t + " pk_qi = pk_i0 + pk_tid;")
  lines.append("      if (pk_qi < pk_m_end) {")
  if isinstance(lse_buf, Var) and lse_off is not None:
    lines.append(f"        pk_sh_lse[pk_tid] = {em.data(lse_buf)}[({idx_t})({em.expr(lse_off)}) + pk_qi];")
  else:
    assert isinstance(m_buf, Var) and m_off is not None and isinstance(l_buf, Var) and l_off is not None
    lines.append(
      f"        pk_sh_lse[pk_tid] = {em.data(m_buf)}[({idx_t})({em.expr(m_off)}) + pk_qi] + "
      f"{log_fn}({em.data(l_buf)}[({idx_t})({em.expr(l_off)}) + pk_qi]);"
    )
  lines.append(f"        pk_sh_g[pk_tid] = {em.data(g_buf)}[({idx_t})({em.expr(g_off)}) + pk_qi];")
  lines.append(f"        {idx_t} pk_tv = ({idx_t}){em.data(t_buf)}[({idx_t})({em.expr(t_off)}) + pk_qi];")
  lines.append("        if (pk_tv < 0) pk_tv += pk_N;")
  lines.append("        pk_sh_tgt[pk_tid] = pk_tv;")
  lines.append("      } else {")
  lines.append(f"        pk_sh_lse[pk_tid] = {zero_lit}; pk_sh_g[pk_tid] = {zero_lit}; pk_sh_tgt[pk_tid] = -1;")
  lines.append("      }")
  lines.append("    }")
  lines.append(f"    {barrier}")

  # 1. Recompute S[v, i] = H[i] . W[:, v]
  lines.append(f"    {ct} pk_s_acc[8];")
  lines.append(f"    for (int pk_c = 0; pk_c < 8; ++pk_c) pk_s_acc[pk_c] = {zero_lit};")
  lines.append(f"    for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {MONOID_BK}) {{")
  _emit_stage_load_2d(
    em,
    plan.stages["h"],
    "pk_sh_a",
    MONOID_BM,
    MONOID_BK,
    k_pad,
    "pk_i0",
    "pk_k0",
    "pk_m_end",
    "pk_K",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  _emit_stage_load_2d(
    em,
    plan.stages["w"],
    "pk_sh_b",
    MONOID_BN,
    MONOID_BK,
    k_pad,
    "pk_j0",
    "pk_k0",
    "pk_N",
    "pk_K",
    ct,
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  lines.append(f"      {barrier}")
  lines.append(f"      const {idx_t} pk_klen = pk_K - pk_k0 < {MONOID_BK} ? pk_K - pk_k0 : {MONOID_BK};")
  lines.append("      for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("        const " + idx_t + " pk_ic = pk_c * 4 + pk_t;")
  lines.append(f"        {ct} pk_dot = pk_s_acc[pk_c];")
  lines.append(f"        for ({idx_t} pk_kk = 0; pk_kk < pk_klen; ++pk_kk) {{")
  lines.append(f"          pk_dot += pk_sh_a[pk_ic * {k_pad} + pk_kk] * pk_sh_b[pk_r * {k_pad} + pk_kk];")
  lines.append("        }")
  lines.append("        pk_s_acc[pk_c] = pk_dot;")
  lines.append("      }")
  lines.append(f"      {barrier}")
  lines.append("    }")

  # 2. dlog[v, i] = g[i] * ((v == tgt[i] ? 1 : 0) - exp(S[v, i] - lse[i]))
  lines.append("    for (int pk_c = 0; pk_c < 8; ++pk_c) {")
  lines.append("      const " + idx_t + " pk_ic = pk_c * 4 + pk_t;")
  lines.append("      const " + idx_t + " pk_gi = pk_i0 + pk_ic;")
  lines.append(f"      {ct} pk_dl = {zero_lit};")
  lines.append("      if (pk_gi < pk_m_end && pk_gv < pk_N) {")
  lines.append(f"        const {ct} pk_pv = {expf}(pk_s_acc[pk_c] - pk_sh_lse[pk_ic]);")
  lines.append(f"        pk_dl = pk_sh_g[pk_ic] * ((pk_gv == pk_sh_tgt[pk_ic] ? {one_lit} : {zero_lit}) - pk_pv);")
  lines.append("      }")
  lines.append(f"      pk_sh_dl[pk_r * {s_pad} + pk_ic] = pk_dl;")
  lines.append("    }")
  lines.append(f"    {barrier}")
  lines.append(f"    const {idx_t} pk_ilen = pk_m_end - pk_i0 < {MONOID_BM} ? pk_m_end - pk_i0 : {MONOID_BM};")
  lines.append(f"    const {idx_t} pk_jlen = pk_N - pk_j0 < {MONOID_BN} ? pk_N - pk_j0 : {MONOID_BN};")

  # 3. Accumulate dW[k, v] += sum_i dlog[v, i] * H[i, k] AND dH[i, k] += atomicAdd(sum_v dlog[v, i] * W[k, v])
  lines.append(f"    for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {MONOID_BK}) {{")
  if act_w:
    _emit_stage_load_2d(
      em,
      plan.stages["h"],
      "pk_sh_a",
      MONOID_BM,
      MONOID_BK,
      k_pad,
      "pk_i0",
      "pk_k0",
      "pk_m_end",
      "pk_K",
      ct,
      idx_t,
      lines,
      "      ",
      zero_lit,
    )
  if act_h:
    _emit_stage_load_2d(
      em,
      plan.stages["w"],
      "pk_sh_b",
      MONOID_BN,
      MONOID_BK,
      k_pad,
      "pk_j0",
      "pk_k0",
      "pk_N",
      "pk_K",
      ct,
      idx_t,
      lines,
      "      ",
      zero_lit,
    )
  lines.append(f"      {barrier}")
  if act_w:
    lines.append("      for (int pk_c = 0; pk_c < 8; ++pk_c) {")
    lines.append("        const " + idx_t + " pk_kc = pk_c * 4 + pk_t;")
    lines.append("        const " + idx_t + " pk_gk = pk_k0 + pk_kc;")
    lines.append("        if (pk_gv < pk_N && pk_gk < pk_K) {")
    lines.append(f"          {ct} pk_acc = {zero_lit};")
    lines.append(f"          for ({idx_t} pk_ii = 0; pk_ii < pk_ilen; ++pk_ii) {{")
    lines.append(f"            pk_acc += pk_sh_dl[pk_r * {s_pad} + pk_ii] * pk_sh_a[pk_ii * {k_pad} + pk_kc];")
    lines.append("          }")
    lines.append(f"          if (pk_k0 < {MAX_REG_D}) {{")
    lines.append("            pk_dw_reg[(int)((pk_k0 >> 2) + pk_c)] += pk_acc;")
    lines.append("          } else {")
    _emit_atomic_add(is_cuda, f"&{em.data(dw_buf)}[pk_gk * pk_N + pk_gv]", "pk_acc", elt, lines, "            ")
    lines.append("          }")
    lines.append("        }")
    lines.append("      }")
  if act_h:
    lines.append("      {")
    lines.append("        const " + idx_t + " pk_qi = pk_i0 + pk_r;")
    lines.append("        if (pk_qi < pk_m_end) {")
    lines.append("          for (int pk_c = 0; pk_c < 8; ++pk_c) {")
    lines.append("            const " + idx_t + " pk_kc = pk_c * 4 + pk_t;")
    lines.append("            const " + idx_t + " pk_gk = pk_k0 + pk_kc;")
    lines.append("            if (pk_gk < pk_K) {")
    lines.append(f"              {ct} pk_dhv = {zero_lit};")
    lines.append(f"              for ({idx_t} pk_jj = 0; pk_jj < pk_jlen; ++pk_jj) {{")
    lines.append(f"                pk_dhv += pk_sh_dl[pk_jj * {s_pad} + pk_r] * pk_sh_b[pk_jj * {k_pad} + pk_kc];")
    lines.append("              }")
    _emit_atomic_add(is_cuda, f"&{em.data(dh_buf)}[pk_qi * pk_K + pk_gk]", "pk_dhv", elt, lines, "              ")
    lines.append("            }")
    lines.append("          }")
    lines.append("        }")
    lines.append("      }")
  lines.append(f"      {barrier}")
  lines.append("    }")
  lines.append("  }")  # end i0 loop

  if act_w:
    lines.append("  if (pk_gv < pk_N) {")
    lines.append(f"    for (int pk_u = 0; pk_u < {MAX_REG_D // 4}; ++pk_u) {{")
    lines.append("      const " + idx_t + " pk_gk = pk_u * 4 + pk_t;")
    lines.append("      if (pk_gk < pk_K) {")
    _emit_atomic_add(is_cuda, f"&{em.data(dw_buf)}[pk_gk * pk_N + pk_gv]", "pk_dw_reg[pk_u]", elt, lines, "        ")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
  lines.append("}")

  written = {b.uid for b in plan.out_bufs.values()}
  grid_shape = [plan.m_dim, plan.n_dim]
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in grid_shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={
      "tiled": {
        "bm": cce_m_chunk,
        "bn": MONOID_BN,
        "bk": MONOID_BK,
        "tm": 1,
        "tn": 1,
        "threads": MONOID_THREADS,
        "shape": shape_meta,
      }
    },
  )


def _use_monoidal_matrix_unit(plan: MonoidalTilePlan, target: str, profile: TargetProfile | None) -> bool:
  """True when ``plan`` can use hardware matrix units (Metal ``32x32x32``
  ``simdgroup_matrix`` or CUDA ``64x32x32`` ``nvcuda::wmma`` TF32 Tensor Cores)."""
  if profile is None or profile.matrix_unit is None:
    return False
  if plan.elt != Float32 or plan.score_op != "dot":
    return False
  if target == "metal":
    return profile.matrix_unit.kind == "simdgroup_matrix"
  if target == "cuda":
    return profile.matrix_unit.kind == "wmma" and profile.tensor_cores
  return False


def _derive_mm_law5_guards(
  plan: MonoidalTilePlan,
  bm: int = MM_BM,
  bn: int = MM_BN,
) -> tuple[ir.Expr | None, ir.Expr | None, tuple[Var, Var] | None]:
  """Re-derive Law 5 tile-interval guards for ``(bm, bn)`` from ``plan.mask_cond``."""
  if plan.mask_cond is None or plan.mask_index_vars is None:
    return None, None, None
  mi_v, mj_v = plan.mask_index_vars
  i0, j0 = plan.tile_origin_vars if plan.tile_origin_vars is not None else (Var("pk_ti0", ops.I), Var("pk_tj0", ops.I))
  iv_map = {
    mi_v.uid: TileInterval(i0, ops.iadd(i0, ir.Const(bm - 1, ops.I))),
    mj_v.uid: TileInterval(j0, ops.iadd(j0, ir.Const(bn - 1, ops.I))),
  }
  af, at = tile_predicate_guards(plan.mask_cond, iv_map, IndexFacts())
  af_out = None if (isinstance(af, ir.Const) and not af.value) else af
  at_out = None if (isinstance(at, ir.Const) and not at.value) else at
  if af_out is None and at_out is None:
    return None, None, None
  return af_out, at_out, (i0, j0)


def _emit_mm_stage_load_2d(
  em: CFamilyEmitter,
  st: TileStage,
  sh_name: str,
  r_size: int,
  c_size: int,
  out_stride: int,
  r0_expr: str,
  c0_expr: str,
  r_lim: str,
  c_lim: str,
  idx_t: str,
  lines: list[str],
  pad: str,
  zero_lit: str,
  transpose: bool = False,
) -> None:
  """Cooperatively load ``r_size x c_size`` tile of ``st`` into shared memory
  (optionally transposed as ``sh[c * out_stride + r]``)."""
  r_shift = r_size.bit_length() - 1
  c_shift = c_size.bit_length() - 1
  d0_fast = _stage_prefers_dim0_fast(st)
  lines.append(f"{pad}for ({idx_t} pk_l = pk_tid; pk_l < {r_size * c_size}; pk_l += {MM_THREADS}) {{")
  if d0_fast:
    lines.append(f"{pad}  const {idx_t} pk_lc = pk_l >> {r_shift};")
    lines.append(f"{pad}  const {idx_t} pk_lr = pk_l & {r_size - 1};")
  else:
    lines.append(f"{pad}  const {idx_t} pk_lr = pk_l >> {c_shift};")
    lines.append(f"{pad}  const {idx_t} pk_lc = pk_l & {c_size - 1};")
  lines.append(f"{pad}  const {idx_t} {st.index[0].name} = {r0_expr} + pk_lr;")
  lines.append(f"{pad}  const {idx_t} {st.index[1].name} = {c0_expr} + pk_lc;")
  sh_idx = f"pk_lc * {out_stride} + pk_lr" if transpose else f"pk_lr * {out_stride} + pk_lc"
  lines.append(f"{pad}  if ({st.index[0].name} < {r_lim} && {st.index[1].name} < {c_lim}) {{")
  _emit_decl_locals(em, st.body, {st.index[0].uid, st.index[1].uid}, lines, pad + "    ")
  em.block(st.body, lines, (len(pad) + 4) // 2)
  lines.append(f"{pad}    {sh_name}[{sh_idx}] = (float)({em.expr(st.value)});")
  lines.append(f"{pad}  }} else {{")
  lines.append(f"{pad}    {sh_name}[{sh_idx}] = {zero_lit};")
  lines.append(f"{pad}  }}")
  lines.append(f"{pad}}}")


def _emit_mm_header(
  em: CFamilyEmitter,
  target: str,
  prelude: str,
  scalars: list[Var],
  bufs: list[Var],
) -> tuple[list[str], str, str, str, str]:
  """Emit kernel header for hardware matrix-unit monoidal kernels."""
  is_cuda = target == "cuda"
  idx_t = "long long" if is_cuda else "long"
  sh_qual = "__shared__" if is_cuda else "threadgroup"
  barrier = "__syncthreads();" if is_cuda else "threadgroup_barrier(mem_flags::mem_threadgroup);"
  warp_barrier = "__syncwarp();" if is_cuda else "simdgroup_barrier(mem_flags::mem_threadgroup);"
  if is_cuda:
    hdr = "#ifndef PK_CUDA_SIMULATE\n#include <mma.h>\n#endif\n" + prelude
  else:
    hdr = prelude.replace("#include <metal_stdlib>", "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>", 1)
  lines: list[str] = [hdr]
  if is_cuda:
    params = ["long long pk_total", "long long pk_csize"]
    params += [f"{em.ctype(v.type)} {v.name}" for v in scalars]
    params += ["int* pk_flag", "long long* pk_info"]
    params += [f"{em.ctype(v.type.elt)}* {v.name}_d" for v in bufs]
    lines.append('extern "C" __global__ void pk_kernel(' + ",\n    ".join(params) + ") {")
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})blockIdx.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})threadIdx.x;")
    lines.append(f"  const {idx_t} pk_sgid = pk_tid >> 5;")
    lines.append(f"  const {idx_t} pk_lane = pk_tid & 31;")
    lines.append(f"  const {idx_t} pk_lr = pk_lane >> 1;")
    lines.append(f"  const {idx_t} pk_lt = pk_lane & 1;")
    lines.append(f"  const {idx_t} pk_r = pk_sgid * 16 + pk_lr;")
  else:
    lines += ["struct pk_args {", "  long total;", "  long csize;"]
    for v in scalars:
      lines.append(f"  {em.ctype(v.type)} {v.name};")
    lines.append("};")
    params = [
      "constant pk_args& pk_a [[buffer(0)]]",
      "device atomic_int* pk_flag [[buffer(1)]]",
      "device long* pk_info [[buffer(2)]]",
    ]
    for k, v in enumerate(bufs):
      params.append(f"device {em.ctype(v.type.elt)}* {v.name}_d [[buffer({k + 3})]]")
    params.append("uint3 pk_bid [[threadgroup_position_in_grid]]")
    params.append("uint3 pk_ltid [[thread_position_in_threadgroup]]")
    params.append("uint pk_sgid_u [[simdgroup_index_in_threadgroup]]")
    params.append("uint pk_lane_u [[thread_index_in_simdgroup]]")
    lines.append("kernel void pk_kernel(" + ",\n    ".join(params) + ") {")
    for v in scalars:
      lines.append(f"  const {em.ctype(v.type)} {v.name} = pk_a.{v.name};")
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})pk_bid.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})pk_ltid.x;")
    lines.append(f"  const {idx_t} pk_sgid = ({idx_t})pk_sgid_u;")
    lines.append(f"  const {idx_t} pk_lane = ({idx_t})pk_lane_u;")
    lines.append(f"  const {idx_t} pk_lr = pk_lane >> 2;")
    lines.append(f"  const {idx_t} pk_lt = pk_lane & 3;")
    lines.append(f"  const {idx_t} pk_r = pk_sgid * 8 + pk_lr;")
  return lines, idx_t, sh_qual, barrier, warp_barrier


def _generate_flash_monoid_matrix_unit(
  loop: ParLoop,
  plan: MonoidalTilePlan,
  free: list[Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None,
) -> KernelPlan:
  """Generate low-register, hoisted-Q hardware matrix-unit FlashAttention kernel
  (Metal ``32x32`` ``simdgroup_matrix`` / CUDA ``64x32`` ``nvcuda::wmma`` TF32)."""
  is_cuda = target == "cuda"
  bm = MM_BM if is_cuda else MM_METAL_BM
  bn = MM_BN if is_cuda else MM_METAL_BN
  bk = MM_BK if is_cuda else MM_METAL_BK
  lanes_per_row = 2 if is_cuda else 4
  cols_per_lane = bn // lanes_per_row
  q_hoist_stride = 68
  k_stride = bk + 4
  s_stride = bn + 4
  b_kt_stride = bn + 4
  b_elems = max(64 * b_kt_stride, bn * q_hoist_stride)
  ep_per_warp = 256 if is_cuda else 64

  zero_lit = "0.0f" if is_cuda else "0.0"
  one_lit = "1.0f" if is_cuda else "1.0"
  neg_inf = "((float)(-INFINITY))" if is_cuda else "(-INFINITY)"
  expf = "expf" if is_cuda else "exp"
  logf = "logf" if is_cuda else "log"
  fmax = "fmaxf" if is_cuda else "fmax"

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  lines, idx_t, sh_qual, barrier, warp_barrier = _emit_mm_header(em, target, prelude, scalars, bufs)
  if is_cuda:
    lines.append("  using namespace nvcuda::wmma;")
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.m_dim)});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.n_dim)});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.k_dim)});")
  assert plan.d_dim is not None
  lines.append(f"  const {idx_t} pk_D = ({idx_t})({em.expr(plan.d_dim)});")
  _emit_batch_unravel(em, plan, "pk_M", bm, idx_t, lines)
  lines.append(f"  const {idx_t} pk_i0 = pk_blk_r * {bm};")
  lines.append(f"  const {idx_t} pk_gi = pk_i0 + pk_r;")
  lines.append(f"  const {idx_t} pk_row_flat = pk_batch_id * pk_M + pk_gi;")
  if plan.score_scale is not None:
    lines.append(f"  const float pk_scale = (float)({em.expr(plan.score_scale)});")
  if plan.mask_fill is not None:
    lines.append(f"  const float pk_mfill = (float)({em.expr(plan.mask_fill)});")

  lines.append(f"  {sh_qual} float pk_sh_a[{bm * q_hoist_stride}];")
  lines.append(f"  {sh_qual} float pk_sh_b[{b_elems}];")
  lines.append(f"  {sh_qual} float pk_sh_s[{bm * s_stride}];")
  lines.append(f"  {sh_qual} float pk_sh_ep[{4 * ep_per_warp}];")
  lines.append(f"  {sh_qual} float pk_sh_m[{bm}];")
  lines.append(f"  {sh_qual} float pk_sh_l[{bm}];")
  lines.append(f"  {sh_qual} float pk_sh_alpha[{bm}];")
  lines.append(f"  {sh_qual} float pk_sh_red[{bm * lanes_per_row}];")

  has_mask_buf = plan.mask_buf is not None and plan.mask_index is not None and plan.mask_fill is not None
  has_mask_cond = plan.mask_cond is not None and plan.mask_fill is not None
  if has_mask_buf:
    lines.append(f"  {sh_qual} uint32_t pk_sh_mask[{bm}];")
    lines.append(f"  {sh_qual} int pk_any_mask;")
    lines.append(f"  {sh_qual} int pk_row_valid[{bm}];")
    lines.append(f"  {sh_qual} int pk_all_valid;")
    lines.append(f"  if (pk_tid < {bm}) pk_row_valid[pk_tid] = 0;")
    lines.append("  if (pk_tid == 0) pk_all_valid = 0;")

  o_buf = plan.out_bufs["o"]
  lines.append("  if (pk_lt == 0) {")
  lines.append(f"    pk_sh_m[pk_r] = {neg_inf};")
  lines.append(f"    pk_sh_l[pk_r] = {zero_lit};")
  lines.append("  }")
  o_reg_cnt = MAX_REG_D // lanes_per_row
  lines.append(f"  float pk_o_reg[{o_reg_cnt}];")
  lines.append(f"  for (int pk_u = 0; pk_u < {o_reg_cnt}; ++pk_u) pk_o_reg[pk_u] = {zero_lit};")
  lines.append(f"  if (pk_gi < pk_M && pk_D > {MAX_REG_D}) {{")
  lines.append(f"    for ({idx_t} pk_d = pk_lt + {MAX_REG_D}; pk_d < pk_D; pk_d += {lanes_per_row}) {{")
  lines.append(f"      {em.data(o_buf)}[pk_row_flat * pk_D + pk_d] = {zero_lit};")
  lines.append("    }")
  lines.append("  }")

  lines.append("  const bool pk_q_hoisted = (pk_K <= 64);")
  lines.append("  if (pk_q_hoisted) {")
  _emit_mm_stage_load_2d(
    em,
    plan.stages["q"],
    "pk_sh_a",
    bm,
    64,
    q_hoist_stride,
    "pk_i0",
    "0",
    "pk_M",
    "pk_K",
    idx_t,
    lines,
    "    ",
    zero_lit,
  )
  lines.append("  }")
  lines.append(f"  {barrier}")

  if "j_start" in plan.aux_exprs and "j_end" in plan.aux_exprs:
    lines.append(f"  const {idx_t} pk_j_start = ({idx_t})({em.expr(plan.aux_exprs['j_start'])});")
    lines.append(f"  const {idx_t} pk_j_end = ({idx_t})({em.expr(plan.aux_exprs['j_end'])});")
  else:
    lines.append(f"  const {idx_t} pk_j_start = 0;")
    lines.append(f"  const {idx_t} pk_j_end = pk_N;")
  is_neg_inf_fill = isinstance(plan.mask_fill, ir.Const) and float(plan.mask_fill.value) == float("-inf")
  skip_eps = "105.0f" if is_cuda else "105.0"
  tile_false_g, tile_true_g, origin_vars = _derive_mm_law5_guards(plan, bm, bn)

  lines.append(f"  for ({idx_t} pk_j0 = pk_j_start; pk_j0 < pk_j_end; pk_j0 += {bn}) {{")
  if tile_false_g is not None and origin_vars is not None:
    i0_v, j0_v = origin_vars
    lines.append(f"    const {idx_t} {i0_v.name} = pk_i0;")
    lines.append(f"    const {idx_t} {j0_v.name} = pk_j0;")
    if is_neg_inf_fill:
      lines.append(f"    if ({em.expr(tile_false_g)}) continue;")
    else:
      lines.append(f"    if ({em.expr(tile_false_g)}) {{")
      lines.append("      int pk_can_skip = 1;")
      lines.append(f"      for ({idx_t} pk_ri = 0; pk_ri < {bm}; ++pk_ri) {{")
      lines.append(
        f"        if (pk_i0 + pk_ri < pk_M && !(pk_sh_l[pk_ri] > {zero_lit} && pk_sh_m[pk_ri] > pk_mfill + {skip_eps})) {{ pk_can_skip = 0; break; }}"
      )
      lines.append("      }")
      lines.append("      if (pk_can_skip) continue;")
      lines.append("    }")
  if tile_true_g is not None and origin_vars is not None:
    i0_v, j0_v = origin_vars
    if tile_false_g is None:
      lines.append(f"    const {idx_t} {i0_v.name} = pk_i0;")
      lines.append(f"    const {idx_t} {j0_v.name} = pk_j0;")
    lines.append(f"    const bool pk_tile_all_true = ({em.expr(tile_true_g)});")
  else:
    lines.append("    const bool pk_tile_all_true = false;")

  if has_mask_buf:
    assert plan.mask_index_vars is not None and plan.mask_buf is not None and plan.mask_index is not None
    mi_v, mj_v = plan.mask_index_vars
    lines.append("    if (pk_tid == 0) pk_any_mask = 0;")
    lines.append(f"    {barrier}")
    lines.append(f"    if (pk_tid < {bm}) {{")
    lines.append(f"      const {idx_t} {mi_v.name} = pk_i0 + pk_tid;")
    lines.append("      uint32_t pk_bits = 0u;")
    lines.append(f"      if ({mi_v.name} < pk_M) {{")
    lines.append("        for (int pk_b = 0; pk_b < 32; ++pk_b) {")
    lines.append(f"          const {idx_t} {mj_v.name} = pk_j0 + pk_b;")
    lines.append(f"          if ({mj_v.name} < pk_j_end && {em.data(plan.mask_buf)}[{em.expr(plan.mask_index)}]) {{")
    lines.append("            pk_bits |= (1u << pk_b);")
    lines.append("          }")
    lines.append("        }")
    lines.append("        if (pk_bits != 0u) { pk_any_mask = 1; pk_row_valid[pk_tid] = 1; }")
    lines.append("      }")
    lines.append("      pk_sh_mask[pk_tid] = pk_bits;")
    lines.append("    }")
    lines.append(f"    {barrier}")
    if is_neg_inf_fill:
      lines.append("    if (pk_any_mask == 0 && pk_all_valid) continue;")
    else:
      lines.append("    if (pk_any_mask == 0 && pk_all_valid) {")
      lines.append("      int pk_can_skip = 1;")
      lines.append(f"      for ({idx_t} pk_ri = 0; pk_ri < {bm}; ++pk_ri) {{")
      lines.append(
        f"        if (pk_i0 + pk_ri < pk_M && !(pk_sh_l[pk_ri] > {zero_lit} && pk_sh_m[pk_ri] > pk_mfill + {skip_eps})) {{ pk_can_skip = 0; break; }}"
      )
      lines.append("      }")
      lines.append("      if (pk_can_skip) continue;")
      lines.append("    }")
    lines.append("    if (pk_tid == 0 && !pk_all_valid) {")
    lines.append("      int pk_av = 1;")
    lines.append(f"      for ({idx_t} pk_ri = 0; pk_ri < {bm}; ++pk_ri) {{")
    lines.append("        if (pk_i0 + pk_ri < pk_M && !pk_row_valid[pk_ri]) { pk_av = 0; break; }")
    lines.append("      }")
    lines.append("      pk_all_valid = pk_av;")
    lines.append("    }")

  if is_cuda:
    lines.append("    fragment<accumulator, 16, 16, 8, float> pk_s_frag[2];")
    lines.append("    fill_fragment(pk_s_frag[0], 0.0f);")
    lines.append("    fill_fragment(pk_s_frag[1], 0.0f);")
  else:
    lines.append("    simdgroup_float8x8 pk_s_frag[4];")
    for rn in range(4):
      lines.append(f"    pk_s_frag[{rn}] = make_filled_simdgroup_matrix<float, 8, 8>(0.0);")

  lines.append("    if (pk_q_hoisted) {")
  _emit_mm_stage_load_2d(
    em,
    plan.stages["k"],
    "pk_sh_b",
    bn,
    64,
    b_kt_stride if is_cuda else q_hoist_stride,
    "pk_j0",
    "0",
    "pk_j_end",
    "pk_K",
    idx_t,
    lines,
    "      ",
    zero_lit,
    transpose=is_cuda,
  )
  lines.append(f"      {barrier}")
  lines.append(f"      for ({idx_t} pk_tk = 0; pk_tk < pk_K; pk_tk += 8) {{")
  if is_cuda:
    lines.append("        fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> pk_fa;")
    lines.append("        fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> pk_fb0, pk_fb1;")
    lines.append(
      f"        load_matrix_sync(pk_fa, &pk_sh_a[(pk_sgid * 16) * {q_hoist_stride} + pk_tk], {q_hoist_stride});"
    )
    lines.append(f"        load_matrix_sync(pk_fb0, &pk_sh_b[pk_tk * {b_kt_stride}], {b_kt_stride});")
    lines.append(f"        load_matrix_sync(pk_fb1, &pk_sh_b[pk_tk * {b_kt_stride} + 16], {b_kt_stride});")
    lines.append("        mma_sync(pk_s_frag[0], pk_fa, pk_fb0, pk_s_frag[0]);")
    lines.append("        mma_sync(pk_s_frag[1], pk_fa, pk_fb1, pk_s_frag[1]);")
  else:
    lines.append("        simdgroup_float8x8 pk_fa;")
    lines.append(
      f"        simdgroup_load(pk_fa, &pk_sh_a[(pk_sgid * 8) * {q_hoist_stride} + pk_tk], {q_hoist_stride});"
    )
    for rn in range(4):
      lines.append("        { simdgroup_float8x8 pk_fb;")
      lines.append(
        f"          simdgroup_load(pk_fb, &pk_sh_b[({rn * 8}) * {q_hoist_stride} + pk_tk], {q_hoist_stride}, ulong2(0, 0), true);"
      )
      lines.append(f"          simdgroup_multiply_accumulate(pk_s_frag[{rn}], pk_fa, pk_fb, pk_s_frag[{rn}]); }}")
  lines.append("      }")
  lines.append("    } else {")
  lines.append(f"      for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {bk}) {{")
  _emit_mm_stage_load_2d(
    em,
    plan.stages["q"],
    "pk_sh_a",
    bm,
    bk,
    k_stride,
    "pk_i0",
    "pk_k0",
    "pk_M",
    "pk_K",
    idx_t,
    lines,
    "        ",
    zero_lit,
  )
  _emit_mm_stage_load_2d(
    em,
    plan.stages["k"],
    "pk_sh_b",
    bn,
    bk,
    b_kt_stride if is_cuda else k_stride,
    "pk_j0",
    "pk_k0",
    "pk_j_end",
    "pk_K",
    idx_t,
    lines,
    "        ",
    zero_lit,
    transpose=is_cuda,
  )
  lines.append(f"        {barrier}")
  lines.append(f"        const {idx_t} pk_klen = pk_K - pk_k0 < {bk} ? pk_K - pk_k0 : {bk};")
  lines.append(f"        for ({idx_t} pk_tk = 0; pk_tk < pk_klen; pk_tk += 8) {{")
  if is_cuda:
    lines.append("          fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> pk_fa;")
    lines.append("          fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> pk_fb0, pk_fb1;")
    lines.append(f"          load_matrix_sync(pk_fa, &pk_sh_a[(pk_sgid * 16) * {k_stride} + pk_tk], {k_stride});")
    lines.append(f"          load_matrix_sync(pk_fb0, &pk_sh_b[pk_tk * {b_kt_stride}], {b_kt_stride});")
    lines.append(f"          load_matrix_sync(pk_fb1, &pk_sh_b[pk_tk * {b_kt_stride} + 16], {b_kt_stride});")
    lines.append("          mma_sync(pk_s_frag[0], pk_fa, pk_fb0, pk_s_frag[0]);")
    lines.append("          mma_sync(pk_s_frag[1], pk_fa, pk_fb1, pk_s_frag[1]);")
  else:
    lines.append("          simdgroup_float8x8 pk_fa;")
    lines.append(f"          simdgroup_load(pk_fa, &pk_sh_a[(pk_sgid * 8) * {k_stride} + pk_tk], {k_stride});")
    for rn in range(4):
      lines.append("          { simdgroup_float8x8 pk_fb;")
      lines.append(
        f"            simdgroup_load(pk_fb, &pk_sh_b[({rn * 8}) * {k_stride} + pk_tk], {k_stride}, ulong2(0, 0), true);"
      )
      lines.append(f"            simdgroup_multiply_accumulate(pk_s_frag[{rn}], pk_fa, pk_fb, pk_s_frag[{rn}]); }}")
  lines.append("        }")
  lines.append(f"        {barrier}")
  lines.append("      }")
  lines.append("    }")

  if is_cuda:
    lines.append(
      f"    store_matrix_sync(&pk_sh_s[(pk_sgid * 16) * {s_stride}], pk_s_frag[0], {s_stride}, mem_row_major);"
    )
    lines.append(
      f"    store_matrix_sync(&pk_sh_s[(pk_sgid * 16) * {s_stride} + 16], pk_s_frag[1], {s_stride}, mem_row_major);"
    )
  else:
    for rn in range(4):
      lines.append(
        f"    simdgroup_store(pk_s_frag[{rn}], &pk_sh_s[(pk_sgid * 8) * {s_stride} + {rn * 8}], {s_stride});"
      )
  lines.append(f"    {warp_barrier}")

  lines.append(f"    float pk_lmax = {neg_inf};")
  lines.append(f"    for (int pk_c = 0; pk_c < {cols_per_lane}; ++pk_c) {{")
  lines.append(f"      const {idx_t} pk_jc = pk_c * {lanes_per_row} + pk_lt;")
  lines.append(f"      const {idx_t} pk_gj = pk_j0 + pk_jc;")
  lines.append(f"      float pk_sv = {neg_inf};")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_j_end) {")
  lines.append(f"        pk_sv = pk_sh_s[pk_r * {s_stride} + pk_jc];")
  if plan.score_scale is not None:
    lines.append("        pk_sv *= pk_scale;")
  if has_mask_buf:
    lines.append("        if (!((pk_sh_mask[pk_r] >> pk_jc) & 1u)) pk_sv = pk_mfill;")
  elif has_mask_cond:
    assert plan.mask_index_vars is not None and plan.mask_cond is not None
    mi_v, mj_v = plan.mask_index_vars
    lines.append("        if (!pk_tile_all_true) {")
    lines.append(f"          const {idx_t} {mi_v.name} = pk_gi;")
    lines.append(f"          const {idx_t} {mj_v.name} = pk_gj;")
    lines.append(f"          if (!({em.expr(plan.mask_cond)})) pk_sv = pk_mfill;")
    lines.append("        }")
  lines.append(f"        pk_lmax = {fmax}(pk_lmax, pk_sv);")
  lines.append("      }")
  lines.append(f"      pk_sh_s[pk_r * {s_stride} + pk_jc] = pk_sv;")
  lines.append("    }")
  lines.append(f"    pk_sh_red[pk_r * {lanes_per_row} + pk_lt] = pk_lmax;")
  lines.append(f"    {warp_barrier}")

  lines.append("    if (pk_lt == 0 && pk_gi < pk_M) {")
  if lanes_per_row == 2:
    lines.append(f"      float pk_tm = {fmax}(pk_sh_red[pk_r * 2], pk_sh_red[pk_r * 2 + 1]);")
  else:
    lines.append(
      f"      float pk_tm = {fmax}({fmax}(pk_sh_red[pk_r * 4], pk_sh_red[pk_r * 4 + 1]), {fmax}(pk_sh_red[pk_r * 4 + 2], pk_sh_red[pk_r * 4 + 3]));"
    )
  lines.append("      float pk_mold = pk_sh_m[pk_r];")
  lines.append(f"      if (pk_j0 == pk_j_start || pk_mold == {neg_inf}) {{")
  lines.append("        pk_sh_m[pk_r] = pk_tm;")
  lines.append(f"        pk_sh_alpha[pk_r] = {zero_lit};")
  lines.append("      } else {")
  lines.append(f"        float pk_mnew = {fmax}(pk_mold, pk_tm);")
  lines.append("        pk_sh_m[pk_r] = pk_mnew;")
  lines.append(f"        pk_sh_alpha[pk_r] = {expf}(pk_mold - pk_mnew);")
  lines.append("      }")
  lines.append(f"    }} else if (pk_lt == 0) {{ pk_sh_alpha[pk_r] = {zero_lit}; }}")
  lines.append(f"    {warp_barrier}")

  lines.append("    const float pk_mcur = pk_sh_m[pk_r];")
  lines.append("    const float pk_al = pk_sh_alpha[pk_r];")
  lines.append(f"    float pk_lsum = {zero_lit};")
  lines.append(f"    for (int pk_c = 0; pk_c < {cols_per_lane}; ++pk_c) {{")
  lines.append(f"      const {idx_t} pk_jc = pk_c * {lanes_per_row} + pk_lt;")
  lines.append(f"      const {idx_t} pk_gj = pk_j0 + pk_jc;")
  lines.append(f"      float pk_pv = {zero_lit};")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_j_end) {")
  lines.append(f"        float pk_sv2 = pk_sh_s[pk_r * {s_stride} + pk_jc];")
  if is_neg_inf_fill:
    lines.append(f"        pk_pv = (pk_sv2 > {neg_inf}) ? {expf}(pk_sv2 - pk_mcur) : {zero_lit};")
  else:
    lines.append(f"        pk_pv = {expf}(pk_sv2 - pk_mcur);")
  lines.append("        pk_lsum += pk_pv;")
  lines.append("      }")
  lines.append(f"      pk_sh_s[pk_r * {s_stride} + pk_jc] = pk_pv;")
  lines.append("    }")
  lines.append(f"    pk_sh_red[pk_r * {lanes_per_row} + pk_lt] = pk_lsum;")
  lines.append(f"    {warp_barrier}")
  lines.append("    if (pk_lt == 0 && pk_gi < pk_M) {")
  if lanes_per_row == 2:
    lines.append("      pk_sh_l[pk_r] = pk_sh_l[pk_r] * pk_al + (pk_sh_red[pk_r * 2] + pk_sh_red[pk_r * 2 + 1]);")
  else:
    lines.append(
      "      pk_sh_l[pk_r] = pk_sh_l[pk_r] * pk_al + (pk_sh_red[pk_r * 4] + pk_sh_red[pk_r * 4 + 1] + pk_sh_red[pk_r * 4 + 2] + pk_sh_red[pk_r * 4 + 3]);"
    )
  lines.append("    }")
  lines.append(f"    {barrier}")

  ep_ptr = (
    f"float* pk_ep = &pk_sh_ep[pk_sgid * {ep_per_warp}];"
    if is_cuda
    else f"threadgroup float* pk_ep = &pk_sh_ep[pk_sgid * {ep_per_warp}];"
  )
  lines.append(f"    {ep_ptr}")
  lines.append(f"    for ({idx_t} pk_d0 = 0; pk_d0 < pk_D; pk_d0 += 64) {{")
  _emit_mm_stage_load_2d(
    em,
    plan.stages["v"],
    "pk_sh_b",
    bn,
    64,
    q_hoist_stride,
    "pk_j0",
    "pk_d0",
    "pk_j_end",
    "pk_D",
    idx_t,
    lines,
    "      ",
    zero_lit,
    transpose=False,
  )
  lines.append(f"      {barrier}")
  if is_cuda:
    lines.append("      for (int pk_rn = 0; pk_rn < 4; ++pk_rn) {")
    lines.append("        if (pk_d0 + pk_rn * 16 >= pk_D) break;")
    lines.append("        fragment<accumulator, 16, 16, 8, float> pk_pv;")
    lines.append("        fill_fragment(pk_pv, 0.0f);")
    lines.append("        #pragma unroll")
    lines.append(f"        for (int pk_tk = 0; pk_tk < {bn}; pk_tk += 8) {{")
    lines.append("          fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> pk_fp;")
    lines.append("          fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> pk_fv;")
    lines.append(f"          load_matrix_sync(pk_fp, &pk_sh_s[(pk_sgid * 16) * {s_stride} + pk_tk], {s_stride});")
    lines.append(
      f"          load_matrix_sync(pk_fv, &pk_sh_b[pk_tk * {q_hoist_stride} + pk_rn * 16], {q_hoist_stride});"
    )
    lines.append("          mma_sync(pk_pv, pk_fp, pk_fv, pk_pv);")
    lines.append("        }")
    lines.append("        store_matrix_sync(pk_ep, pk_pv, 16, mem_row_major);")
    lines.append(f"        {warp_barrier}")
    lines.append("        for (int pk_u = 0; pk_u < 8; ++pk_u) {")
    lines.append(f"          const {idx_t} pk_col16 = pk_u * 2 + pk_lt;")
    lines.append(f"          const {idx_t} pk_gd = pk_d0 + pk_rn * 16 + pk_col16;")
    lines.append("          if (pk_gi < pk_M && pk_gd < pk_D) {")
    lines.append("            const float pk_pv_sum = pk_ep[pk_lr * 16 + pk_col16];")
    lines.append(f"            if (pk_d0 < {MAX_REG_D}) {{")
    lines.append("              const int pk_slot = (int)((pk_d0 >> 1) + pk_rn * 8 + pk_u);")
    lines.append("              pk_o_reg[pk_slot] = pk_o_reg[pk_slot] * pk_al + pk_pv_sum;")
    lines.append("            } else {")
    lines.append(
      f"              {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] = {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] * pk_al + pk_pv_sum;"
    )
    lines.append("            }")
    lines.append("          }")
    lines.append("        }")
    lines.append(f"        {warp_barrier}")
    lines.append("      }")
  else:
    lines.append("      for (int pk_rn = 0; pk_rn < 8; ++pk_rn) {")
    lines.append("        if (pk_d0 + pk_rn * 8 >= pk_D) break;")
    lines.append("        simdgroup_float8x8 pk_pv = make_filled_simdgroup_matrix<float, 8, 8>(0.0);")
    lines.append("        #pragma unroll")
    lines.append(f"        for (int pk_tk = 0; pk_tk < {bn}; pk_tk += 8) {{")
    lines.append("          simdgroup_float8x8 pk_fp, pk_fv;")
    lines.append(f"          simdgroup_load(pk_fp, &pk_sh_s[(pk_sgid * 8) * {s_stride} + pk_tk], {s_stride});")
    lines.append(f"          simdgroup_load(pk_fv, &pk_sh_b[pk_tk * {q_hoist_stride} + pk_rn * 8], {q_hoist_stride});")
    lines.append("          simdgroup_multiply_accumulate(pk_pv, pk_fp, pk_fv, pk_pv);")
    lines.append("        }")
    lines.append("        simdgroup_store(pk_pv, pk_ep, 8);")
    lines.append(f"        {warp_barrier}")
    lines.append("        for (int pk_u = 0; pk_u < 2; ++pk_u) {")
    lines.append(f"          const {idx_t} pk_col8 = pk_u * 4 + pk_lt;")
    lines.append(f"          const {idx_t} pk_gd = pk_d0 + pk_rn * 8 + pk_col8;")
    lines.append("          if (pk_gi < pk_M && pk_gd < pk_D) {")
    lines.append("            const float pk_pv_sum = pk_ep[pk_lr * 8 + pk_col8];")
    lines.append(f"            if (pk_d0 < {MAX_REG_D}) {{")
    lines.append("              const int pk_slot = (int)((pk_d0 >> 2) + pk_rn * 2 + pk_u);")
    lines.append("              pk_o_reg[pk_slot] = pk_o_reg[pk_slot] * pk_al + pk_pv_sum;")
    lines.append("            } else {")
    lines.append(
      f"              {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] = {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] * pk_al + pk_pv_sum;"
    )
    lines.append("            }")
    lines.append("          }")
    lines.append("        }")
    lines.append(f"        {warp_barrier}")
    lines.append("      }")
  lines.append(f"      {barrier}")
  lines.append("    }")
  lines.append("  }")

  unnorm = "unnormalized" in plan.aux_exprs
  lines.append("  if (pk_gi < pk_M) {")
  if unnorm:
    lines.append(f"    const float pk_inv_l = {one_lit};")
  else:
    lines.append(f"    const float pk_inv_l = {one_lit} / pk_sh_l[pk_r];")
  lines.append(f"    for (int pk_u = 0; pk_u < {o_reg_cnt}; ++pk_u) {{")
  lines.append(f"      const {idx_t} pk_gd = pk_u * {lanes_per_row} + pk_lt;")
  lines.append("      if (pk_gd < pk_D) {")
  lines.append(f"        {em.data(o_buf)}[pk_row_flat * pk_D + pk_gd] = pk_o_reg[pk_u] * pk_inv_l;")
  lines.append("      }")
  lines.append("    }")
  lines.append(f"    for ({idx_t} pk_d = pk_lt + {MAX_REG_D}; pk_d < pk_D; pk_d += {lanes_per_row}) {{")
  lines.append(f"      {em.data(o_buf)}[pk_row_flat * pk_D + pk_d] *= pk_inv_l;")
  lines.append("    }")
  lines.append("    if (pk_lt == 0) {")
  if "m" in plan.out_bufs:
    lines.append(f"      {em.data(plan.out_bufs['m'])}[pk_row_flat] = pk_sh_m[pk_r];")
  if "l" in plan.out_bufs:
    lines.append(f"      {em.data(plan.out_bufs['l'])}[pk_row_flat] = pk_sh_l[pk_r];")
  if "lse" in plan.out_bufs:
    lines.append(f"      {em.data(plan.out_bufs['lse'])}[pk_row_flat] = pk_sh_m[pk_r] + {logf}(pk_sh_l[pk_r]);")
  lines.append("    }")
  lines.append("  }")
  lines.append("}")

  written = {b.uid for b in plan.out_bufs.values()}
  grid_shape = [*plan.batch_shape, plan.m_dim, ir.Const(1, ops.I)]
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in grid_shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={
      "tiled": {
        "bm": bm,
        "bn": 1,
        "bk": bk,
        "tm": 1,
        "tn": 1,
        "threads": MM_THREADS,
        "matrix_unit": "wmma" if is_cuda else "simdgroup_matrix",
        "shape": shape_meta,
      }
    },
  )


def _generate_monoid_lse_matrix_unit(
  loop: ParLoop,
  plan: MonoidalTilePlan,
  free: list[Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None,
) -> KernelPlan:
  """Generate low-register hardware matrix-unit Cut Cross-Entropy / LogSumExp kernel
  (Metal ``32x32x32`` ``simdgroup_matrix`` / CUDA ``64x32x32`` ``nvcuda::wmma`` TF32)."""
  is_cuda = target == "cuda"
  bm = MM_BM if is_cuda else MM_METAL_BM
  bn = MM_BN if is_cuda else MM_METAL_BN
  bk = MM_BK if is_cuda else MM_METAL_BK
  lanes_per_row = 2 if is_cuda else 4
  cols_per_lane = bn // lanes_per_row
  k_stride = bk + 4
  s_stride = bn + 4
  b_kt_stride = bn + 4

  zero_lit = "0.0f" if is_cuda else "0.0"
  neg_inf = "((float)(-INFINITY))" if is_cuda else "(-INFINITY)"
  expf = "expf" if is_cuda else "exp"
  logf = "logf" if is_cuda else "log"
  fmax = "fmaxf" if is_cuda else "fmax"

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  lines, idx_t, sh_qual, barrier, warp_barrier = _emit_mm_header(em, target, prelude, scalars, bufs)
  if is_cuda:
    lines.append("  using namespace nvcuda::wmma;")
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.m_dim)});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.n_dim)});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.k_dim)});")
  _emit_batch_unravel(em, plan, "pk_M", bm, idx_t, lines)
  lines.append(f"  const {idx_t} pk_i0 = pk_blk_r * {bm};")
  lines.append(f"  const {idx_t} pk_gi = pk_i0 + pk_r;")
  lines.append(f"  const {idx_t} pk_row_flat = pk_batch_id * pk_M + pk_gi;")
  if plan.score_scale is not None:
    lines.append(f"  const float pk_scale = (float)({em.expr(plan.score_scale)});")

  lines.append(f"  {sh_qual} float pk_sh_a[{bm * k_stride}];")
  lines.append(f"  {sh_qual} float pk_sh_b[{bk * b_kt_stride if is_cuda else bn * k_stride}];")
  lines.append(f"  {sh_qual} float pk_sh_s[{bm * s_stride}];")
  lines.append(f"  {sh_qual} float pk_sh_m[{bm}];")
  lines.append(f"  {sh_qual} float pk_sh_l[{bm}];")
  lines.append(f"  {sh_qual} float pk_sh_red[{bm * lanes_per_row}];")

  has_target = "target_buf" in plan.aux_exprs and "target_index" in plan.aux_exprs
  if has_target:
    tbuf = plan.aux_exprs["target_buf"]
    tidx = plan.aux_exprs["target_index"]
    assert isinstance(tbuf, Var)
    lines.append(f"  {sh_qual} {idx_t} pk_sh_tgt[{bm}];")
    lines.append(f"  {sh_qual} float pk_sh_pick[{bm}];")
    lines.append("  if (pk_lt == 0) {")
    lines.append(f"    pk_sh_pick[pk_r] = {zero_lit};")
    lines.append("    if (pk_gi < pk_M) {")
    t_orig = next((x for x in ir.walk_expr(tidx) if isinstance(x, Var) and x.name.startswith("lo_")), None)
    if t_orig is not None:
      lines.append(f"      const {idx_t} {t_orig.name} = pk_gi;")
    lines.append(f"      {idx_t} pk_tv = ({idx_t}){em.data(tbuf)}[{em.expr(tidx)}];")
    lines.append("      if (pk_tv < 0) pk_tv += pk_N;")
    lines.append("      pk_sh_tgt[pk_r] = pk_tv;")
    lines.append("    } else {")
    lines.append("      pk_sh_tgt[pk_r] = -1;")
    lines.append("    }")
    lines.append("  }")

  lines.append("  if (pk_lt == 0) {")
  lines.append(f"    pk_sh_m[pk_r] = {neg_inf};")
  lines.append(f"    pk_sh_l[pk_r] = {zero_lit};")
  lines.append("  }")
  lines.append(f"  {barrier}")

  lines.append(f"  for ({idx_t} pk_j0 = 0; pk_j0 < pk_N; pk_j0 += {bn}) {{")
  if is_cuda:
    lines.append("    fragment<accumulator, 16, 16, 8, float> pk_s_frag[2];")
    lines.append("    fill_fragment(pk_s_frag[0], 0.0f);")
    lines.append("    fill_fragment(pk_s_frag[1], 0.0f);")
  else:
    lines.append("    simdgroup_float8x8 pk_s_frag[4];")
    for rn in range(4):
      lines.append(f"    pk_s_frag[{rn}] = make_filled_simdgroup_matrix<float, 8, 8>(0.0);")

  lines.append(f"    for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {bk}) {{")
  _emit_mm_stage_load_2d(
    em,
    plan.stages["h"],
    "pk_sh_a",
    bm,
    bk,
    k_stride,
    "pk_i0",
    "pk_k0",
    "pk_M",
    "pk_K",
    idx_t,
    lines,
    "      ",
    zero_lit,
  )
  _emit_mm_stage_load_2d(
    em,
    plan.stages["w"],
    "pk_sh_b",
    bn,
    bk,
    b_kt_stride if is_cuda else k_stride,
    "pk_j0",
    "pk_k0",
    "pk_N",
    "pk_K",
    idx_t,
    lines,
    "      ",
    zero_lit,
    transpose=is_cuda,
  )
  lines.append(f"      {barrier}")
  lines.append(f"      const {idx_t} pk_klen = pk_K - pk_k0 < {bk} ? pk_K - pk_k0 : {bk};")
  lines.append(f"      for ({idx_t} pk_tk = 0; pk_tk < pk_klen; pk_tk += 8) {{")
  if is_cuda:
    lines.append("        fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> pk_fa;")
    lines.append("        fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> pk_fb0, pk_fb1;")
    lines.append(f"        load_matrix_sync(pk_fa, &pk_sh_a[(pk_sgid * 16) * {k_stride} + pk_tk], {k_stride});")
    lines.append(f"        load_matrix_sync(pk_fb0, &pk_sh_b[pk_tk * {b_kt_stride}], {b_kt_stride});")
    lines.append(f"        load_matrix_sync(pk_fb1, &pk_sh_b[pk_tk * {b_kt_stride} + 16], {b_kt_stride});")
    lines.append("        mma_sync(pk_s_frag[0], pk_fa, pk_fb0, pk_s_frag[0]);")
    lines.append("        mma_sync(pk_s_frag[1], pk_fa, pk_fb1, pk_s_frag[1]);")
  else:
    lines.append("        simdgroup_float8x8 pk_fa;")
    lines.append(f"        simdgroup_load(pk_fa, &pk_sh_a[(pk_sgid * 8) * {k_stride} + pk_tk], {k_stride});")
    for rn in range(4):
      lines.append("        { simdgroup_float8x8 pk_fb;")
      lines.append(
        f"          simdgroup_load(pk_fb, &pk_sh_b[({rn * 8}) * {k_stride} + pk_tk], {k_stride}, ulong2(0, 0), true);"
      )
      lines.append(f"          simdgroup_multiply_accumulate(pk_s_frag[{rn}], pk_fa, pk_fb, pk_s_frag[{rn}]); }}")
  lines.append("      }")
  lines.append(f"      {barrier}")
  lines.append("    }")

  if is_cuda:
    lines.append(
      f"    store_matrix_sync(&pk_sh_s[(pk_sgid * 16) * {s_stride}], pk_s_frag[0], {s_stride}, mem_row_major);"
    )
    lines.append(
      f"    store_matrix_sync(&pk_sh_s[(pk_sgid * 16) * {s_stride} + 16], pk_s_frag[1], {s_stride}, mem_row_major);"
    )
  else:
    for rn in range(4):
      lines.append(
        f"    simdgroup_store(pk_s_frag[{rn}], &pk_sh_s[(pk_sgid * 8) * {s_stride} + {rn * 8}], {s_stride});"
      )
  lines.append(f"    {warp_barrier}")

  lines.append(f"    float pk_lmax = {neg_inf};")
  lines.append(f"    for (int pk_c = 0; pk_c < {cols_per_lane}; ++pk_c) {{")
  lines.append(f"      const {idx_t} pk_jc = pk_c * {lanes_per_row} + pk_lt;")
  lines.append(f"      const {idx_t} pk_gj = pk_j0 + pk_jc;")
  lines.append(f"      float pk_sv = {neg_inf};")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_N) {")
  lines.append(f"        pk_sv = pk_sh_s[pk_r * {s_stride} + pk_jc];")
  if plan.score_scale is not None:
    lines.append("        pk_sv *= pk_scale;")
  if has_target:
    lines.append("        if (pk_gj == pk_sh_tgt[pk_r]) pk_sh_pick[pk_r] = pk_sv;")
  lines.append(f"        pk_lmax = {fmax}(pk_lmax, pk_sv);")
  lines.append("      }")
  lines.append(f"      pk_sh_s[pk_r * {s_stride} + pk_jc] = pk_sv;")
  lines.append("    }")
  lines.append(f"    pk_sh_red[pk_r * {lanes_per_row} + pk_lt] = pk_lmax;")
  lines.append(f"    {warp_barrier}")

  lines.append("    if (pk_lt == 0 && pk_gi < pk_M) {")
  if lanes_per_row == 2:
    lines.append(f"      float pk_tm = {fmax}(pk_sh_red[pk_r * 2], pk_sh_red[pk_r * 2 + 1]);")
  else:
    lines.append(
      f"      float pk_tm = {fmax}({fmax}(pk_sh_red[pk_r * 4], pk_sh_red[pk_r * 4 + 1]), {fmax}(pk_sh_red[pk_r * 4 + 2], pk_sh_red[pk_r * 4 + 3]));"
    )
  lines.append("      if (pk_j0 == 0) {")
  lines.append("        pk_sh_m[pk_r] = pk_tm;")
  lines.append("      } else {")
  lines.append("        float pk_mold = pk_sh_m[pk_r];")
  lines.append(f"        float pk_mnew = {fmax}(pk_mold, pk_tm);")
  lines.append("        pk_sh_m[pk_r] = pk_mnew;")
  lines.append(f"        pk_sh_l[pk_r] *= {expf}(pk_mold - pk_mnew);")
  lines.append("      }")
  lines.append("    }")
  lines.append(f"    {warp_barrier}")

  lines.append("    const float pk_mcur = pk_sh_m[pk_r];")
  lines.append(f"    float pk_lsum = {zero_lit};")
  lines.append(f"    for (int pk_c = 0; pk_c < {cols_per_lane}; ++pk_c) {{")
  lines.append(f"      const {idx_t} pk_jc = pk_c * {lanes_per_row} + pk_lt;")
  lines.append(f"      const {idx_t} pk_gj = pk_j0 + pk_jc;")
  lines.append("      if (pk_gi < pk_M && pk_gj < pk_N) {")
  lines.append(f"        pk_lsum += {expf}(pk_sh_s[pk_r * {s_stride} + pk_jc] - pk_mcur);")
  lines.append("      }")
  lines.append("    }")
  lines.append(f"    pk_sh_red[pk_r * {lanes_per_row} + pk_lt] = pk_lsum;")
  lines.append(f"    {warp_barrier}")
  lines.append("    if (pk_lt == 0 && pk_gi < pk_M) {")
  if lanes_per_row == 2:
    lines.append("      pk_sh_l[pk_r] += (pk_sh_red[pk_r * 2] + pk_sh_red[pk_r * 2 + 1]);")
  else:
    lines.append(
      "      pk_sh_l[pk_r] += (pk_sh_red[pk_r * 4] + pk_sh_red[pk_r * 4 + 1] + pk_sh_red[pk_r * 4 + 2] + pk_sh_red[pk_r * 4 + 3]);"
    )
  lines.append("    }")
  lines.append(f"    {barrier}")
  lines.append("  }")

  lines.append("  if (pk_lt == 0 && pk_gi < pk_M) {")
  lines.append(f"    const float pk_lse = pk_sh_m[pk_r] + {logf}(pk_sh_l[pk_r]);")
  if "m" in plan.out_bufs:
    lines.append(f"    {em.data(plan.out_bufs['m'])}[pk_row_flat] = pk_sh_m[pk_r];")
  if "l" in plan.out_bufs:
    lines.append(f"    {em.data(plan.out_bufs['l'])}[pk_row_flat] = pk_sh_l[pk_r];")
  if "out" in plan.out_bufs and has_target:
    lines.append(f"    {em.data(plan.out_bufs['out'])}[pk_row_flat] = pk_sh_pick[pk_r] - pk_lse;")
  if "picked_sub_m" in plan.out_bufs and has_target:
    lines.append(f"    {em.data(plan.out_bufs['picked_sub_m'])}[pk_row_flat] = pk_sh_pick[pk_r] - pk_sh_m[pk_r];")
  if "lse" in plan.out_bufs:
    lines.append(f"    {em.data(plan.out_bufs['lse'])}[pk_row_flat] = pk_lse;")
  lines.append("  }")
  lines.append("}")

  written = {b.uid for b in plan.out_bufs.values()}
  grid_shape = [*plan.batch_shape, plan.m_dim, ir.Const(1, ops.I)]
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in grid_shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={
      "tiled": {
        "bm": bm,
        "bn": 1,
        "bk": bk,
        "tm": 1,
        "tn": 1,
        "threads": MM_THREADS,
        "matrix_unit": "wmma" if is_cuda else "simdgroup_matrix",
        "shape": shape_meta,
      }
    },
  )


def generate_monoidal_gpu_kernel(
  loop: ParLoop,
  free: list[Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None = None,
  profile: TargetProfile | None = None,
) -> KernelPlan | None:
  """Generate a cooperative SRAM-tiled CUDA or Metal kernel for ``loop.monoidal_tile``."""
  plan = loop.monoidal_tile
  if plan is None:
    return None
  use_mm = _use_monoidal_matrix_unit(plan, target, profile)
  if plan.kind == "flash_monoid":
    if use_mm:
      return _generate_flash_monoid_matrix_unit(loop, plan, free, target, em, prelude, max_buffers)
    return _generate_flash_monoid(loop, plan, free, target, em, prelude, max_buffers)
  if plan.kind == "flash_monoid_bwd":
    return _generate_flash_monoid_bwd(loop, plan, free, target, em, prelude, max_buffers)
  if plan.kind == "monoid_lse":
    if use_mm:
      return _generate_monoid_lse_matrix_unit(loop, plan, free, target, em, prelude, max_buffers)
    return _generate_monoid_lse(loop, plan, free, target, em, prelude, max_buffers)
  if plan.kind == "cce_bwd":
    return _generate_cce_bwd(loop, plan, free, target, em, prelude, max_buffers)
  return None


__all__ = [
  "MM_BK",
  "MM_BM",
  "MM_BN",
  "MM_THREADS",
  "MONOID_BK",
  "MONOID_BM",
  "MONOID_BN",
  "MONOID_THREADS",
  "generate_monoidal_gpu_kernel",
]
