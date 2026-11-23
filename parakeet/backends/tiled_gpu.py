"""Shared GPU block-tiled kernel generator (CUDA ``__shared__`` & Metal ``threadgroup``).

Given a :class:`~parakeet.syntax.ir.SharedTilePlan` attached to a lowered
:class:`~parakeet.syntax.ir.ParLoop`, this module analytically sizes the
threadblock and register micro-tile ``(BM, BN, BK, TM, TN)`` from the target
SRAM/register budget and generates a bank-conflict-free, coalesced,
register-blocked GPU kernel supporting:

* arbitrary batch rank ``(B..., M, N)`` via a 1D block grid,
* arbitrary number of staged row/col/k prologue expressions (``a @ b``,
  ``sin(a) @ cos(b)``, ``allpairs``, multi-input contractions),
* arbitrary folds (seeded or unseeded, associative or custom ``combine``,
  multiple folds, and arbitrary ``Payload``s), and
* arbitrary fused epilogues (``np.tanh(a @ b + c)``, ``sqrt(...) + 1.0``).
"""

from __future__ import annotations

import math

from ..ndtypes import BufType, ScalarType
from ..syntax import ir
from .cfamily import CFamilyEmitter, Unsupported, defined_vars, expr_types, used_vars
from .device import KernelPlan

BLOCK_THREADS = 256  # 16 x 16 threads per block / threadgroup
NTX = 16
NTY = 16
BK = 16
PAD = BK + 1  # +1 padding eliminates shared-memory bank conflicts (gcd(17, 32) = 1)
DEFAULT_SHARED_BUDGET = 24 * 1024  # 24 KiB target -> high occupancy on CUDA & Metal


def choose_tile_config(
  plan: ir.SharedTilePlan, max_shared_bytes: int = DEFAULT_SHARED_BUDGET
) -> tuple[int, int, int, int, int] | None:
  """Analytically choose ``(BM, BN, BK, TM, TN)`` from ``plan``'s stage types,
  accumulator count, and target shared-memory budget."""
  n_accs = sum(1 + len(f.payloads) for f in plan.mixed_folds)
  candidates: list[tuple[int, int, int, int, int]] = []
  if n_accs <= 2:
    candidates.append((64, 64, BK, 4, 4))
  if n_accs <= 8:
    candidates.append((32, 32, BK, 2, 2))
  candidates.append((16, 16, BK, 1, 1))

  for bm, bn, bk, tm, tn in candidates:
    needed = 0
    for st, side in zip(plan.stages, plan.stage_sides):
      elt_bytes = max(st.buf.type.elt.bits // 8, 1)
      if side == "row":
        needed += bm * (bk + 1) * elt_bytes
      elif side == "col":
        needed += bn * (bk + 1) * elt_bytes
      else:
        needed += bk * elt_bytes
    if needed <= max_shared_bytes:
      return bm, bn, bk, tm, tn
  return None


def tiled_num_blocks(tiled_meta: dict, ctx) -> int:
  """Compute the 1D threadblock grid size for a tiled GPU launch from ``ctx``."""
  dims = [val if kind == "c" else int(getattr(ctx, val)) for kind, val in tiled_meta["shape"]]
  if any(d <= 0 for d in dims):
    return 0
  bm, bn = tiled_meta["bm"], tiled_meta["bn"]
  batch = math.prod(dims[:-2]) if len(dims) > 2 else 1
  return batch * ((dims[-2] + bm - 1) // bm) * ((dims[-1] + bn - 1) // bn)


def _zero_literal(t: ScalarType) -> str:
  return "false" if t.kind == "b" else "0"


def _collect_epilogue_before(before: list[ir.Stmt], after: list[ir.Stmt]) -> list[ir.Stmt]:
  """Return the minimal prefix of ``before`` assigns still needed by ``after``."""
  assigns = [s for s in before if isinstance(s, ir.Assign)]
  by_uid = {s.var.uid: s for s in assigns}
  order = {s.var.uid: k for k, s in enumerate(assigns)}
  need: set[int] = set()

  def visit_expr(e: ir.Expr) -> None:
    for x in ir.walk_expr(e):
      if isinstance(x, ir.Var) and x.uid in by_uid and x.uid not in need:
        need.add(x.uid)
        visit_expr(by_uid[x.uid].value)

  for s in ir.walk_stmts(after):
    for e in ir.stmt_exprs(s):
      visit_expr(e)
  return [by_uid[u] for u in sorted(need, key=lambda u: order[u])]


def _plan_types(plan: ir.SharedTilePlan) -> set[ScalarType]:
  types: set[ScalarType] = set()
  for e in (*plan.shape, *plan.reduce.shape):
    for x in ir.walk_expr(e):
      if isinstance(x.type, ScalarType):
        types.add(x.type)
  for st in plan.stages:
    types.add(st.buf.type.elt)
    types |= expr_types(st.body)
    for x in ir.walk_expr(st.value):
      if isinstance(x.type, ScalarType):
        types.add(x.type)
  types |= expr_types(plan.mixed_body)
  types |= expr_types(plan.before)
  types |= expr_types(plan.after)
  for f in plan.mixed_folds:
    for v in (f.result, f.a, f.b):
      if isinstance(v.type, ScalarType):
        types.add(v.type)
    for e in (f.value, f.combined, *(a for a in (f.init,) if a is not None)):
      for x in ir.walk_expr(e):
        if isinstance(x.type, ScalarType):
          types.add(x.type)
    types |= expr_types(f.combine)
    for p in f.payloads:
      for v in (p.result, p.a, p.b):
        if isinstance(v.type, ScalarType):
          types.add(v.type)
      for e in (p.value, p.combined):
        for x in ir.walk_expr(e):
          if isinstance(x.type, ScalarType):
            types.add(x.type)
  # Collect any uniform pre-checks' types as well
  for chk in plan.pre_checks:
    for e in (chk.cond, *chk.args):
      for x in ir.walk_expr(e):
        if isinstance(x.type, ScalarType):
          types.add(x.type)
  return types


def _emit_decl_locals(em: CFamilyEmitter, stmts: list[ir.Stmt], skip_uids: set[int], out: list[str], pad: str) -> None:
  locs = defined_vars(stmts, into_parloops=True)
  for uid, v in sorted(locs.items()):
    if uid in skip_uids:
      continue
    if isinstance(v.type, BufType):
      raise Unsupported("it creates arrays inside the tiled GPU loop")
    out.append(f"{pad}{em.ctype(v.type)} {v.name} = {_zero_literal(v.type)};")


def _emit_fold_step(
  em: CFamilyEmitter,
  f: ir.Fold,
  acc_name: str,
  pacc_names: list[str],
  val_expr: str,
  pval_exprs: list[str],
  out: list[str],
  pad: str,
) -> None:
  """Emit one accumulation step updating ``acc_name`` (and any payload accumulators)."""
  if not f.combine and not f.payloads and f.op is not None:
    if f.op == "add" and f.a.type.kind == "f":
      out.append(f"{pad}{acc_name} += ({em.ctype(f.a.type)})({val_expr});")
      return
    if f.op == "mul" and f.a.type.kind == "f":
      out.append(f"{pad}{acc_name} *= ({em.ctype(f.a.type)})({val_expr});")
      return
    a_tmp = ir.Var("pk_fa", f.a.type)
    b_tmp = ir.Var("pk_fv", f.b.type)
    out.append(f"{pad}{{ const {em.ctype(f.a.type)} {a_tmp.name} = {acc_name};")
    out.append(f"{pad}  const {em.ctype(f.b.type)} {b_tmp.name} = ({em.ctype(f.b.type)})({val_expr});")
    comb_expr = em.expr(ir.Prim(f.op, (a_tmp, b_tmp), f.a.type))
    out.append(f"{pad}  {acc_name} = {comb_expr}; }}")
    return

  # General combine block and/or multi-payload fold
  out.append(f"{pad}{{")
  out.append(f"{pad}  const {em.ctype(f.a.type)} {f.a.name} = {acc_name};")
  out.append(f"{pad}  const {em.ctype(f.b.type)} {f.b.name} = ({em.ctype(f.b.type)})({val_expr});")
  for p, pacc, pval in zip(f.payloads, pacc_names, pval_exprs):
    out.append(f"{pad}  const {em.ctype(p.a.type)} {p.a.name} = {pacc};")
    out.append(f"{pad}  const {em.ctype(p.b.type)} {p.b.name} = ({pval});")
  skip = {f.a.uid, f.b.uid} | {p.a.uid for p in f.payloads} | {p.b.uid for p in f.payloads}
  _emit_decl_locals(em, f.combine, skip, out, pad + "  ")
  em.block(f.combine, out, len(pad) // 2 + 1)
  out.append(f"{pad}  {acc_name} = {em.expr(f.combined)};")
  for p, pacc in zip(f.payloads, pacc_names):
    out.append(f"{pad}  {pacc} = {em.expr(p.combined)};")
  out.append(f"{pad}}}")


def generate_tiled_gpu_kernel(
  loop: ir.ParLoop,
  free: list[ir.Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None = None,
) -> KernelPlan | None:
  """Generate a cooperative ``__shared__`` (``target="cuda"``) or ``threadgroup``
  (``target="metal"``) block-tiled GPU kernel for ``loop.shared_tile``.

  Returns ``None`` if the plan exceeds SRAM limits (so the caller can fall back
  to the ordinary 1D GPU loop), or raises :class:`Unsupported` if the body uses
  types/ops unavailable on ``target`` (e.g. ``float64`` on Metal).
  """
  plan = loop.shared_tile
  if plan is None:
    return None
  cfg = choose_tile_config(plan)
  if cfg is None:
    return None
  bm, bn, bk, tm, tn = cfg
  pad_k = bk + 1

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")

  # Validate all types up front (rejects float64 on Metal cleanly)
  for t in _plan_types(plan):
    em.ctype(t)
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)

  is_cuda = target == "cuda"
  idx_t = "long long" if is_cuda else "long"
  sh_qual = "__shared__" if is_cuda else "threadgroup"
  barrier = "__syncthreads();" if is_cuda else "threadgroup_barrier(mem_flags::mem_threadgroup);"
  fail_zero = "0LL" if is_cuda else "0L"

  # Keep uniforms largest-first on Metal (matching MetalPlanner's arg_struct layout)
  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  lines: list[str] = [prelude]
  if is_cuda:
    params = ["long long pk_total", "long long pk_csize"]
    params += [f"{em.ctype(v.type)} {v.name}" for v in scalars]
    params += ["int* pk_flag", "long long* pk_info"]
    params += [f"{em.ctype(v.type.elt)}* {v.name}_d" for v in bufs]
    lines.append('extern "C" __global__ void pk_kernel(' + ",\n    ".join(params) + ") {")
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

  # Block / thread coordinates
  if is_cuda:
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})blockIdx.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})threadIdx.x;")
  else:
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})pk_bid.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})pk_ltid.x;")
  lines.append(f"  const {idx_t} pk_tx = pk_tid & {NTX - 1};")
  lines.append(f"  const {idx_t} pk_ty = pk_tid >> 4;")

  # Shape dimensions & block unraveling (supports arbitrary batch rank >= 0)
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.shape[-2])});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.shape[-1])});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.reduce.shape[0])});")
  lines.append(f"  const {idx_t} pk_nb_n = (pk_N + {bn - 1}) / {bn};")
  lines.append(f"  const {idx_t} pk_nb_m = (pk_M + {bm - 1}) / {bm};")
  lines.append(f"  const {idx_t} pk_blk_n = pk_blk % pk_nb_n;")
  lines.append(f"  const {idx_t} pk_rem0 = pk_blk / pk_nb_n;")
  lines.append(f"  const {idx_t} pk_blk_m = pk_rem0 % pk_nb_m;")
  if len(plan.shape) > 2:
    lines.append(f"  {idx_t} pk_bflat = pk_rem0 / pk_nb_m;")
    batch_dims = plan.shape[:-2]
    batch_vars = plan.index[:-2]
    for b_idx in range(len(batch_dims) - 1, 0, -1):
      d_expr = em.expr(batch_dims[b_idx])
      bv = batch_vars[b_idx]
      lines.append(f"  const {idx_t} {bv.name} = pk_bflat % ({idx_t})({d_expr});")
      lines.append(f"  pk_bflat = pk_bflat / ({idx_t})({d_expr});")
    lines.append(f"  const {idx_t} {batch_vars[0].name} = pk_bflat;")
  lines.append(f"  const {idx_t} pk_i0 = pk_blk_m * {bm};")
  lines.append(f"  const {idx_t} pk_j0 = pk_blk_n * {bn};")

  # Uniform pre-checks from m.before (e.g. broadcast dimension checks in allpairs)
  for chk in plan.pre_checks:
    eid = em.error_id(chk.err)
    cargs = [f"({idx_t})({em.expr(a)})" for a in chk.args[:4]]
    cargs += [fail_zero] * (4 - len(cargs))
    lines.append(f"  if (!({em.expr(chk.cond)})) {{")
    lines.append("    if (pk_M > 0 && pk_N > 0 && pk_tid == 0) {")
    lines.append(f"      pk_fail(pk_flag, pk_info, {eid}, {', '.join(cargs)});")
    lines.append("    }")
    lines.append("    return;")
    lines.append("  }")

  # Empty reduction check (uniform across all threads in the block -> deadlock-free)
  empty_err = next((f.empty for f in plan.mixed_folds if f.init is None and f.empty is not None), None)
  if empty_err is not None:
    eid = em.error_id(empty_err)
    lines.append("  if (pk_K <= 0) {")
    lines.append("    if (pk_M > 0 && pk_N > 0 && pk_tid == 0) {")
    lines.append(f"      pk_fail(pk_flag, pk_info, {eid}, {fail_zero}, {fail_zero}, {fail_zero}, {fail_zero});")
    lines.append("    }")
    lines.append("    return;")
    lines.append("  }")

  # Shared / threadgroup memory declarations (padded by +1 on BK to avoid bank conflicts)
  for s_idx, (st, side) in enumerate(zip(plan.stages, plan.stage_sides)):
    ct = em.ctype(st.buf.type.elt)
    if side == "row":
      size = bm * pad_k
    elif side == "col":
      size = bn * pad_k
    else:
      size = bk
    lines.append(f"  {sh_qual} {ct} pk_sh_{s_idx}[{size}];")

  # Register accumulators for TM x TN micro-tile across all folds (and any payloads)
  for f_idx, f in enumerate(plan.mixed_folds):
    ct = em.ctype(f.a.type)
    init_expr = em.expr(f.init) if f.init is not None else _zero_literal(f.a.type)
    for mi in range(tm):
      for nj in range(tn):
        lines.append(f"  {ct} pk_acc_{f_idx}_{mi}_{nj} = ({ct})({init_expr});")
        for p_idx, p in enumerate(f.payloads):
          pct = em.ctype(p.a.type)
          lines.append(f"  {pct} pk_pacc_{f_idx}_{p_idx}_{mi}_{nj} = {_zero_literal(p.a.type)};")

  # Inspect which coordinate vars mixed_body / mixed_folds reference
  mixed_used = used_vars(plan.mixed_body)
  for f in plan.mixed_folds:
    for e in (f.value, *(p.value for p in f.payloads)):
      for x in ir.walk_expr(e):
        if isinstance(x, ir.Var):
          mixed_used[x.uid] = x
  i_var, j_var = plan.index[-2], plan.index[-1]
  k_var = plan.reduce.index[0]
  mixed_uses_i = i_var.uid in mixed_used
  mixed_uses_j = j_var.uid in mixed_used
  mixed_uses_k = k_var.uid in mixed_used
  stage_var_uids = {sv.uid for sv in plan.stage_vars}

  # Main K-tile loop
  lines.append(f"  for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {bk}) {{")

  # Phase 1: cooperative load of all staged operands / prologues
  for s_idx, (st, side) in enumerate(zip(plan.stages, plan.stage_sides)):
    ct = em.ctype(st.buf.type.elt)
    zero = _zero_literal(st.buf.type.elt)
    if side == "row":
      total_elts = bm * bk
      lines.append(f"    for ({idx_t} pk_idx = pk_tid; pk_idx < {total_elts}; pk_idx += {BLOCK_THREADS}) {{")
      lines.append(f"      const {idx_t} pk_ti = pk_idx >> 4;")
      lines.append(f"      const {idx_t} pk_tk = pk_idx & {bk - 1};")
      lines.append(f"      const {idx_t} {st.index[0].name} = pk_i0 + pk_ti;")
      lines.append(f"      const {idx_t} {st.index[1].name} = pk_k0 + pk_tk;")
      lines.append(f"      if ({st.index[0].name} < pk_M && {st.index[1].name} < pk_K) {{")
      _emit_decl_locals(em, st.body, {st.index[0].uid, st.index[1].uid}, lines, "        ")
      em.block(st.body, lines, 4)
      lines.append(f"        pk_sh_{s_idx}[pk_ti * {pad_k} + pk_tk] = ({ct})({em.expr(st.value)});")
      lines.append("      } else {")
      lines.append(f"        pk_sh_{s_idx}[pk_ti * {pad_k} + pk_tk] = {zero};")
      lines.append("      }")
      lines.append("    }")
    elif side == "col":
      total_elts = bn * bk
      lines.append(f"    for ({idx_t} pk_idx = pk_tid; pk_idx < {total_elts}; pk_idx += {BLOCK_THREADS}) {{")
      lines.append(f"      const {idx_t} pk_tj = pk_idx >> 4;")
      lines.append(f"      const {idx_t} pk_tk = pk_idx & {bk - 1};")
      lines.append(f"      const {idx_t} {st.index[0].name} = pk_j0 + pk_tj;")
      lines.append(f"      const {idx_t} {st.index[1].name} = pk_k0 + pk_tk;")
      lines.append(f"      if ({st.index[0].name} < pk_N && {st.index[1].name} < pk_K) {{")
      _emit_decl_locals(em, st.body, {st.index[0].uid, st.index[1].uid}, lines, "        ")
      em.block(st.body, lines, 4)
      lines.append(f"        pk_sh_{s_idx}[pk_tj * {pad_k} + pk_tk] = ({ct})({em.expr(st.value)});")
      lines.append("      } else {")
      lines.append(f"        pk_sh_{s_idx}[pk_tj * {pad_k} + pk_tk] = {zero};")
      lines.append("      }")
      lines.append("    }")
    else:
      lines.append(f"    if (pk_tid < {bk}) {{")
      lines.append(f"      const {idx_t} {st.index[0].name} = pk_k0 + pk_tid;")
      lines.append(f"      if ({st.index[0].name} < pk_K) {{")
      _emit_decl_locals(em, st.body, {st.index[0].uid}, lines, "        ")
      em.block(st.body, lines, 4)
      lines.append(f"        pk_sh_{s_idx}[pk_tid] = ({ct})({em.expr(st.value)});")
      lines.append("      } else {")
      lines.append(f"        pk_sh_{s_idx}[pk_tid] = {zero};")
      lines.append("      }")
      lines.append("    }")

  lines.append(f"    {barrier}")

  # Phase 2: register-blocked TM x TN computation over the staged tile
  lines.append(f"    const {idx_t} pk_klim = (pk_K - pk_k0 < {bk}) ? (pk_K - pk_k0) : {bk};")
  lines.append("    #pragma unroll")
  lines.append(f"    for ({idx_t} pk_tk = 0; pk_tk < pk_klim; pk_tk++) {{")
  if mixed_uses_k:
    lines.append(f"      const {idx_t} {k_var.name} = pk_k0 + pk_tk;")
  # Preload row/col/k shared values for this tk into registers
  for s_idx, (st, side) in enumerate(zip(plan.stages, plan.stage_sides)):
    ct = em.ctype(st.buf.type.elt)
    if side == "row":
      for mi in range(tm):
        lines.append(f"      const {ct} pk_r_{s_idx}_{mi} = pk_sh_{s_idx}[(pk_ty + {mi * NTY}) * {pad_k} + pk_tk];")
    elif side == "col":
      for nj in range(tn):
        lines.append(f"      const {ct} pk_c_{s_idx}_{nj} = pk_sh_{s_idx}[(pk_tx + {nj * NTX}) * {pad_k} + pk_tk];")
    else:
      lines.append(f"      const {ct} pk_k_{s_idx} = pk_sh_{s_idx}[pk_tk];")

  for mi in range(tm):
    for nj in range(tn):
      lines.append("      {")
      pad = "        "
      if plan.has_mixed_loads or mixed_uses_i:
        lines.append(f"{pad}const {idx_t} {i_var.name} = pk_i0 + pk_ty + {mi * NTY};")
      if plan.has_mixed_loads or mixed_uses_j:
        lines.append(f"{pad}const {idx_t} {j_var.name} = pk_j0 + pk_tx + {nj * NTX};")
      if plan.has_mixed_loads:
        lines.append(f"{pad}if ({i_var.name} < pk_M && {j_var.name} < pk_N) {{")
        pad += "  "
      for s_idx, (sv, side) in enumerate(zip(plan.stage_vars, plan.stage_sides)):
        ct = em.ctype(sv.type)
        if side == "row":
          src = f"pk_r_{s_idx}_{mi}"
        elif side == "col":
          src = f"pk_c_{s_idx}_{nj}"
        else:
          src = f"pk_k_{s_idx}"
        lines.append(f"{pad}const {ct} {sv.name} = {src};")
      _emit_decl_locals(em, plan.mixed_body, stage_var_uids | {i_var.uid, j_var.uid, k_var.uid}, lines, pad)
      em.block(plan.mixed_body, lines, len(pad) // 2)
      for f_idx, f in enumerate(plan.mixed_folds):
        acc_name = f"pk_acc_{f_idx}_{mi}_{nj}"
        pacc_names = [f"pk_pacc_{f_idx}_{p_idx}_{mi}_{nj}" for p_idx in range(len(f.payloads))]
        val_expr = em.expr(f.value)
        pval_exprs = [em.expr(p.value) for p in f.payloads]
        if f.init is None:
          lines.append(f"{pad}if (pk_k0 == 0 && pk_tk == 0) {{")
          lines.append(f"{pad}  {acc_name} = ({em.ctype(f.a.type)})({val_expr});")
          for p, pacc, pval in zip(f.payloads, pacc_names, pval_exprs):
            lines.append(f"{pad}  {pacc} = ({em.ctype(p.a.type)})({pval});")
          lines.append(f"{pad}}} else {{")
          _emit_fold_step(em, f, acc_name, pacc_names, val_expr, pval_exprs, lines, pad + "  ")
          lines.append(f"{pad}}}")
        else:
          _emit_fold_step(em, f, acc_name, pacc_names, val_expr, pval_exprs, lines, pad)
      if plan.has_mixed_loads:
        lines.append("        }")
      lines.append("      }")

  lines.append("    }")
  lines.append(f"    {barrier}")
  lines.append("  }")

  # Epilogue: bind fold results + needed setup from plan.before, run plan.after
  epilogue_before = _collect_epilogue_before(plan.before, plan.after)
  epilogue_stmts = epilogue_before + plan.after
  fold_result_uids = {f.result.uid for f in plan.mixed_folds} | {
    p.result.uid for f in plan.mixed_folds for p in f.payloads
  }
  for mi in range(tm):
    for nj in range(tn):
      lines.append("  {")
      lines.append(f"    const {idx_t} {i_var.name} = pk_i0 + pk_ty + {mi * NTY};")
      lines.append(f"    const {idx_t} {j_var.name} = pk_j0 + pk_tx + {nj * NTX};")
      lines.append(f"    if ({i_var.name} < pk_M && {j_var.name} < pk_N) {{")
      for f_idx, f in enumerate(plan.mixed_folds):
        lines.append(f"      const {em.ctype(f.result.type)} {f.result.name} = pk_acc_{f_idx}_{mi}_{nj};")
        for p_idx, p in enumerate(f.payloads):
          lines.append(f"      const {em.ctype(p.result.type)} {p.result.name} = pk_pacc_{f_idx}_{p_idx}_{mi}_{nj};")
      _emit_decl_locals(em, epilogue_stmts, fold_result_uids | {i_var.uid, j_var.uid}, lines, "      ")
      em.block(epilogue_stmts, lines, 3)
      lines.append("    }")
      lines.append("  }")

  lines.append("}")

  written = {s.buf.uid for s in ir.walk_stmts(plan.after) if isinstance(s, ir.Store)}
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in plan.shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={"tiled": {"bm": bm, "bn": bn, "bk": bk, "tm": tm, "tn": tn, "shape": shape_meta}},
  )


__all__ = [
  "BK",
  "BLOCK_THREADS",
  "DEFAULT_SHARED_BUDGET",
  "choose_tile_config",
  "generate_tiled_gpu_kernel",
  "tiled_num_blocks",
]
