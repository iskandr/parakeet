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
  multiple folds, and arbitrary ``Carry``s), and
* arbitrary fused epilogues (``np.tanh(a @ b + c)``, ``sqrt(...) + 1.0``).
"""

from __future__ import annotations

import math

from ..ndtypes import BufType, Float32, ScalarType
from ..syntax import ir
from ..target import BK, DEFAULT_SHARED_BUDGET, TargetProfile, choose_tile_config
from .cfamily import CFamilyEmitter, Unsupported, defined_vars, expr_types, used_vars
from .device import KernelPlan

BLOCK_THREADS = 256  # 16 x 16 threads per block / threadgroup for general SIMT tiled kernels
MATRIX_UNIT_THREADS = 128  # 4 warps / SIMD-groups (2 x 2) for simdgroup_matrix & wmma kernels
NTX = 16
NTY = 16
PAD = BK + 1  # +1 padding eliminates shared-memory bank conflicts (gcd(17, 32) = 1)


def _stage_prefers_dim0_fast(st: ir.TileStage) -> bool:
  """Return True when a 2D ``TileStage(index=(d0, d1))`` loads memory with unit
  stride along ``d0`` rather than ``d1`` (e.g., an un-transposed C-contiguous
  ``B[k, j]`` where ``d0 = j, d1 = k``).  Mapping adjacent GPU threads along the
  unit-stride axis ensures 100% warp-coalesced HBM loads with zero transpose copy."""
  if len(st.index) != 2:
    return False
  d0, d1 = st.index
  local_defs = {t.var.uid: t.value for t in st.body if isinstance(t, ir.Assign)}

  def has_unit(e: ir.Expr, target_uid: int, depth: int = 8) -> bool:
    if depth <= 0:
      return False
    if isinstance(e, ir.Var):
      if e.uid == target_uid:
        return True
      if e.uid in local_defs:
        return has_unit(local_defs[e.uid], target_uid, depth - 1)
      return False
    if isinstance(e, ir.Prim):
      if e.op in ("add", "sub"):
        return any(has_unit(a, target_uid, depth - 1) for a in e.args)
      if e.op == "mul" and len(e.args) == 2:
        a, b = e.args
        if isinstance(a, ir.Const) and a.value == 1:
          return has_unit(b, target_uid, depth - 1)
        if isinstance(b, ir.Const) and b.value == 1:
          return has_unit(a, target_uid, depth - 1)
    return False

  unit0 = unit1 = False
  for expr in (*[e for t in st.body for e in ir.stmt_exprs(t)], st.value):
    for x in ir.walk_expr(expr):
      if isinstance(x, ir.Load):
        if has_unit(x.index, d0.uid):
          unit0 = True
        if has_unit(x.index, d1.uid):
          unit1 = True
  return unit0 and not unit1


def match_matrix_unit_plan(plan: ir.SharedTilePlan) -> tuple[int, int, ir.Fold, float] | None:
  """Return ``(row_stage_idx, col_stage_idx, fold, init_float)`` when ``plan``
  is a float32 sum-of-products contraction eligible for hardware matrix-unit
  tiling (Metal ``simdgroup_matrix`` or CUDA ``nvcuda::wmma``)."""
  if plan.has_mixed_loads or len(plan.stages) != 2 or len(plan.mixed_folds) != 1:
    return None
  if set(plan.stage_sides) != {"row", "col"}:
    return None
  row_idx = plan.stage_sides.index("row")
  col_idx = plan.stage_sides.index("col")
  if plan.stages[row_idx].buf.type.elt != Float32 or plan.stages[col_idx].buf.type.elt != Float32:
    return None
  f = plan.mixed_folds[0]
  if f.op != "add" or f.combine or f.carries or f.a.type != Float32:
    return None
  if not isinstance(f.init, ir.Const):
    return None
  # Resolve straight-line Assigns inside mixed_body and confirm f.value == row_sv * col_sv
  defs: dict[int, ir.Expr] = {}
  for s in plan.mixed_body:
    if isinstance(s, ir.Comment):
      continue
    if not isinstance(s, ir.Assign):
      return None
    defs[s.var.uid] = s.value

  def unwrap(e: ir.Expr, depth: int = 8) -> ir.Expr:
    while depth > 0:
      if isinstance(e, ir.Cast) and e.type == Float32:
        e = e.arg
      elif isinstance(e, ir.Var) and e.uid in defs:
        e = defs[e.uid]
      else:
        break
      depth -= 1
    return e

  val = unwrap(f.value)
  if not isinstance(val, ir.Prim) or val.op != "mul" or len(val.args) != 2:
    return None
  a = unwrap(val.args[0])
  b = unwrap(val.args[1])
  if not (isinstance(a, ir.Var) and isinstance(b, ir.Var)):
    return None
  want = {plan.stage_vars[row_idx].uid, plan.stage_vars[col_idx].uid}
  if {a.uid, b.uid} != want:
    return None
  return row_idx, col_idx, f, float(f.init.value)


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
    for c in f.carries:
      for v in (c.result, c.a, c.b):
        if isinstance(v.type, ScalarType):
          types.add(v.type)
      for e in (c.value, c.combined):
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
  """Emit one accumulation step updating ``acc_name`` (and any carried accumulators)."""
  if not f.combine and not f.carries and f.op is not None:
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

  # General combine block and/or multi-carry fold
  out.append(f"{pad}{{")
  out.append(f"{pad}  const {em.ctype(f.a.type)} {f.a.name} = {acc_name};")
  out.append(f"{pad}  const {em.ctype(f.b.type)} {f.b.name} = ({em.ctype(f.b.type)})({val_expr});")
  for c, pacc, pval in zip(f.carries, pacc_names, pval_exprs):
    out.append(f"{pad}  const {em.ctype(c.a.type)} {c.a.name} = {pacc};")
    out.append(f"{pad}  const {em.ctype(c.b.type)} {c.b.name} = ({pval});")
  skip = {f.a.uid, f.b.uid} | {c.a.uid for c in f.carries} | {c.b.uid for c in f.carries}
  _emit_decl_locals(em, f.combine, skip, out, pad + "  ")
  em.block(f.combine, out, len(pad) // 2 + 1)
  out.append(f"{pad}  {acc_name} = {em.expr(f.combined)};")
  for c, pacc in zip(f.carries, pacc_names):
    out.append(f"{pad}  {pacc} = {em.expr(c.combined)};")
  out.append(f"{pad}}}")


def _match_pure_store_epilogue(plan: ir.SharedTilePlan, fold: ir.Fold) -> tuple[list[ir.Stmt], ir.Store] | None:
  """Return ``(index_assigns, store)`` when ``plan``'s epilogue is a pure
  linear ``out[idx] = fold.result`` store into a float32 buffer (eligible for
  Split-K WMMA accumulation via ``atomicAdd``)."""
  epilogue_before = _collect_epilogue_before(plan.before, plan.after)
  stmts = [s for s in epilogue_before + plan.after if not isinstance(s, ir.Comment)]
  if not stmts or not isinstance(stmts[-1], ir.Store):
    return None
  st = stmts[-1]
  if st.atomic is not None or st.buf.type.elt != Float32:
    return None
  if any(not isinstance(s, ir.Assign) for s in stmts[:-1]):
    return None
  defs = {s.var.uid: s.value for s in stmts[:-1] if isinstance(s, ir.Assign)}
  v: ir.Expr = st.value
  for _ in range(8):
    if isinstance(v, ir.Cast) and v.type == Float32:
      v = v.arg
    elif isinstance(v, ir.Var) and v.uid in defs:
      v = defs[v.uid]
    else:
      break
  if not (isinstance(v, ir.Var) and v.uid == fold.result.uid):
    return None
  idx_uids = {x.uid for x in ir.walk_expr(st.index) if isinstance(x, ir.Var)}
  if fold.result.uid in idx_uids:
    return None
  return stmts[:-1], st


def _choose_cuda_wmma_config(plan: ir.SharedTilePlan) -> tuple[int, int, int, int, int, int, int, int]:
  """Select a shape-adapted CUDA WMMA block/warp/register tile configuration
  ``(bm, bn, bk, wm, wn, r_m, r_n, threads)`` whose double-buffered shared
  memory fits inside 48 KiB static SMEM (2+ blocks/SM on Ampere/Ada)."""
  m_c = int(plan.shape[-2].value) if isinstance(plan.shape[-2], ir.Const) else None
  n_c = int(plan.shape[-1].value) if isinstance(plan.shape[-1], ir.Const) else None
  if (m_c is not None and m_c <= 64) or (n_c is not None and n_c <= 64):
    # 64x64x32 double-buffered: 4 warps (2x2), 2x2 WMMA/warp, 35.0 KiB SMEM
    return 64, 64, 32, 2, 2, 2, 2, 128
  if n_c is not None and n_c <= 96:
    # 128x64x16 double-buffered: 4 warps (2x2), 4x2 WMMA/warp, 28.5 KiB SMEM
    return 128, 64, 16, 2, 2, 4, 2, 128
  if m_c is not None and m_c <= 96:
    # 64x128x16 double-buffered: 4 warps (2x2), 2x4 WMMA/warp, 28.5 KiB SMEM
    return 64, 128, 16, 2, 2, 2, 4, 128
  # Default / large / dynamic: 128x128x16 double-buffered: 8 warps (4x2), 2x4 WMMA/warp, 36.5 KiB SMEM
  return 128, 128, 16, 4, 2, 2, 4, 256


def _extract_unit_stride_load(st: ir.TileStage, fast_var: ir.Var) -> tuple[ir.Var, list[ir.Stmt], ir.Expr] | None:
  """If ``st`` is a pure float32 global load whose index has exact coefficient +1
  along ``fast_var`` (enabling 128-bit ``cp.async`` DMA), return
  ``(buf, index_prep_stmts, raw_index_expr)``; else ``None``."""
  if len(st.index) != 2 or st.buf.type.elt != Float32:
    return None
  local_defs: dict[int, ir.Expr] = {}
  load_assign_uid: int | None = None
  for s in st.body:
    if isinstance(s, ir.Comment):
      continue
    if not isinstance(s, ir.Assign):
      return None
    local_defs[s.var.uid] = s.value

  val: ir.Expr = st.value
  for _ in range(8):
    if isinstance(val, ir.Cast) and val.type == Float32:
      val = val.arg
    elif isinstance(val, ir.Var) and val.uid in local_defs:
      nxt = local_defs[val.uid]
      if isinstance(nxt, ir.Load):
        load_assign_uid = val.uid
      val = nxt
    else:
      break
  if not isinstance(val, ir.Load) or val.buf.type.elt != Float32:
    return None
  raw_idx = val.index
  prep_stmts = [s for s in st.body if isinstance(s, ir.Assign) and s.var.uid != load_assign_uid]
  # Reject any other Load inside the stage (e.g. indirect gather)
  if any(isinstance(x, ir.Load) for s in prep_stmts for x in ir.walk_expr(s.value)):
    return None
  if any(isinstance(x, ir.Load) for x in ir.walk_expr(raw_idx)):
    return None

  fast_uid = fast_var.uid

  def mentions_fast(e: ir.Expr, depth: int = 12) -> bool:
    if depth <= 0:
      return True
    for x in ir.walk_expr(e):
      if isinstance(x, ir.Var):
        if x.uid == fast_uid:
          return True
        if x.uid in local_defs and mentions_fast(local_defs[x.uid], depth - 1):
          return True
    return False

  def coeff_of_fast(e: ir.Expr, depth: int = 12) -> tuple[bool, int]:
    if depth <= 0:
      return False, 0
    if isinstance(e, ir.Const):
      return True, 0
    if isinstance(e, ir.Var):
      if e.uid == fast_uid:
        return True, 1
      if e.uid in local_defs:
        return coeff_of_fast(local_defs[e.uid], depth - 1)
      return True, 0
    if isinstance(e, ir.Prim):
      if e.op == "add" and len(e.args) == 2:
        ok0, c0 = coeff_of_fast(e.args[0], depth - 1)
        ok1, c1 = coeff_of_fast(e.args[1], depth - 1)
        return ok0 and ok1, c0 + c1
      if e.op == "sub" and len(e.args) == 2:
        ok0, c0 = coeff_of_fast(e.args[0], depth - 1)
        ok1, c1 = coeff_of_fast(e.args[1], depth - 1)
        return ok0 and ok1, c0 - c1
      if e.op == "mul" and len(e.args) == 2:
        a, b = e.args
        ok0, c0 = coeff_of_fast(a, depth - 1)
        ok1, c1 = coeff_of_fast(b, depth - 1)
        if ok0 and ok1 and c0 == 0 and c1 == 0:
          return True, 0
        if ok0 and c0 == 0 and isinstance(a, ir.Const) and a.value == 1:
          return ok1, c1
        if ok1 and c1 == 0 and isinstance(b, ir.Const) and b.value == 1:
          return ok0, c0
        return False, 0
    if not mentions_fast(e, depth):
      return True, 0
    return False, 0

  ok, c = coeff_of_fast(raw_idx)
  if not ok or c != 1:
    return None
  return val.buf, prep_stmts, raw_idx


def _emit_cuda_wmma_stage_load(
  em: CFamilyEmitter,
  st: ir.TileStage,
  sh_name: str,
  buf_expr: str,
  slow_dim: int,
  fast_dim: int,
  sh_stride: int,
  slow_var: ir.Var,
  fast_var: ir.Var,
  slow_base: str,
  fast_base: str,
  slow_lim: str,
  fast_lim: str,
  threads: int,
  unit_load: tuple[ir.Var, list[ir.Stmt], ir.Expr] | None,
  lines: list[str],
  pad: str,
) -> None:
  """Emit cooperative loading of one A or B tile into ``sh_name[buf_expr]``,
  using 128-bit ``cp.async`` DMA when ``unit_load`` is contiguous."""
  if unit_load is not None and fast_dim % 4 == 0:
    u_buf, u_prep, u_idx = unit_load
    chunks_per_row = fast_dim // 4
    chunks_shift = chunks_per_row.bit_length() - 1
    total_chunks = slow_dim * chunks_per_row
    b_ptr = em.data(u_buf)
    lines.append(f"{pad}for (long long pk_c = pk_tid; pk_c < {total_chunks}; pk_c += {threads}) {{")
    lines.append(f"{pad}  const long long pk_ts = pk_c >> {chunks_shift};")
    lines.append(f"{pad}  const long long pk_tf = (pk_c & {chunks_per_row - 1}) << 2;")
    lines.append(f"{pad}  const long long {slow_var.name} = {slow_base} + pk_ts;")
    lines.append(f"{pad}  const long long {fast_var.name} = {fast_base} + pk_tf;")
    lines.append(f"{pad}  float* pk_sp = &{sh_name}[{buf_expr}][pk_ts * {sh_stride} + pk_tf];")
    lines.append(f"{pad}  if ({slow_var.name} < {slow_lim}) {{")
    _emit_decl_locals(em, u_prep, {slow_var.uid, fast_var.uid}, lines, pad + "    ")
    em.block(u_prep, lines, len(pad) // 2 + 2)
    lines.append(f"{pad}    const long long pk_goff = (long long)({em.expr(u_idx)});")
    lines.append(f"{pad}    if (({fast_var.name} + 3) < {fast_lim}) {{")
    lines.append(f"{pad}      const float* pk_gp = &{b_ptr}[pk_goff];")
    lines.append(f"{pad}      if ((((unsigned long long)pk_gp) & 15ULL) == 0ULL) {{")
    lines.append(f"{pad}        pk_cp_async_16(pk_sp, pk_gp);")
    lines.append(f"{pad}      }} else {{")
    lines.append(f"{pad}        #pragma unroll")
    lines.append(f"{pad}        for (int pk_v = 0; pk_v < 4; ++pk_v) pk_sp[pk_v] = pk_gp[pk_v];")
    lines.append(f"{pad}      }}")
    lines.append(f"{pad}    }} else {{")
    lines.append(f"{pad}      #pragma unroll")
    lines.append(f"{pad}      for (int pk_v = 0; pk_v < 4; ++pk_v)")
    lines.append(
      f"{pad}        pk_sp[pk_v] = (({fast_var.name} + pk_v) < {fast_lim}) ? {b_ptr}[pk_goff + pk_v] : 0.0f;"
    )
    lines.append(f"{pad}    }}")
    lines.append(f"{pad}  }} else {{")
    lines.append(f"{pad}    #pragma unroll")
    lines.append(f"{pad}    for (int pk_v = 0; pk_v < 4; ++pk_v) pk_sp[pk_v] = 0.0f;")
    lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")
    return

  fast_shift = fast_dim.bit_length() - 1
  lines.append(f"{pad}for (long long pk_idx = pk_tid; pk_idx < {slow_dim * fast_dim}; pk_idx += {threads}) {{")
  lines.append(f"{pad}  const long long pk_ts = pk_idx >> {fast_shift};")
  lines.append(f"{pad}  const long long pk_tf = pk_idx & {fast_dim - 1};")
  lines.append(f"{pad}  const long long {slow_var.name} = {slow_base} + pk_ts;")
  lines.append(f"{pad}  const long long {fast_var.name} = {fast_base} + pk_tf;")
  lines.append(f"{pad}  if ({slow_var.name} < {slow_lim} && {fast_var.name} < {fast_lim}) {{")
  _emit_decl_locals(em, st.body, {slow_var.uid, fast_var.uid}, lines, pad + "    ")
  em.block(st.body, lines, len(pad) // 2 + 2)
  lines.append(f"{pad}    {sh_name}[{buf_expr}][pk_ts * {sh_stride} + pk_tf] = (float)({em.expr(st.value)});")
  lines.append(f"{pad}  }} else {{")
  lines.append(f"{pad}    {sh_name}[{buf_expr}][pk_ts * {sh_stride} + pk_tf] = 0.0f;")
  lines.append(f"{pad}  }}")
  lines.append(f"{pad}}}")


def _generate_matrix_unit_kernel(
  loop: ir.ParLoop,
  plan: ir.SharedTilePlan,
  free: list[ir.Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  row_idx: int,
  col_idx: int,
  fold: ir.Fold,
  init_val: float,
  max_buffers: int | None = None,
) -> KernelPlan:
  """Generate a hardware matrix-unit block-tiled kernel:

  * ``target == "metal"``: Apple Silicon ``simdgroup_matrix<float, 8, 8>`` (full
    IEEE FP32, 128 threads = 4 SIMD-groups, ``64x64x16`` block, 10 KiB SRAM).
  * ``target == "cuda"``: NVIDIA Ampere+/Ada ``nvcuda::wmma`` TF32 Tensor Cores
    with shape-adapted tiles (``128x128x16`` 8-warp default, ``128x64x16``,
    ``64x128x16``, or ``64x64x32``), native ``row_major`` / ``col_major``
    operand layouts (all four NN/NT/TN/TT orientations coalesced), and
    double-buffered 128-bit ``cp.async`` shared-memory pipelining.

  Both support arbitrary batch rank, arbitrary fused prologue stages (``sin(a) @
  cos(b)``), non-multiple-of-tile tail bounds, and arbitrary fused epilogues
  (``np.tanh(a @ b + c)``) with zero extra shared memory.
  """
  is_cuda = target == "cuda"
  st_row = plan.stages[row_idx]
  st_col = plan.stages[col_idx]
  row_d0_fast = _stage_prefers_dim0_fast(st_row)
  col_d0_fast = _stage_prefers_dim0_fast(st_col)

  if is_cuda:
    bm, bn, bk, wm, wn, r_m, r_n, threads = _choose_cuda_wmma_config(plan)
    # Match SMEM orientation to each operand's unit-stride axis so all four
    # NN/NT/TN/TT GEMMs stream contiguously with 128-bit cp.async DMA.
    a_col_major = row_d0_fast
    b_col_major = not col_d0_fast
    a_slow, a_fast = (bk, bm) if a_col_major else (bm, bk)
    b_slow, b_fast = (bn, bk) if b_col_major else (bk, bn)
    a_stride = a_fast + 4  # 16-byte aligned + bank-conflict-free
    b_stride = b_fast + 4
    a_size = a_slow * a_stride
    b_size = b_slow * b_stride
    warp_m = bm // wm
    warp_n = bn // wn
    a_unit = _extract_unit_stride_load(st_row, st_row.index[0] if a_col_major else st_row.index[1])
    b_unit = _extract_unit_stride_load(st_col, st_col.index[1] if b_col_major else st_col.index[0])
  else:
    bm, bn, bk = 64, 64, 16
    wm, wn, r_m, r_n, threads = 2, 2, 4, 4, MATRIX_UNIT_THREADS
    a_col_major = False
    b_col_major = True
    a_stride = bk + 4
    b_stride = bk + 4
    a_size = bm * a_stride
    b_size = bn * b_stride
    warp_m = 32
    warp_n = 32
    a_unit = None
    b_unit = None

  bk_shift = bk.bit_length() - 1
  idx_t = "long long" if is_cuda else "long"
  sh_qual = "__shared__" if is_cuda else "threadgroup"
  barrier = "__syncthreads();" if is_cuda else "threadgroup_barrier(mem_flags::mem_threadgroup);"
  fail_zero = "0LL" if is_cuda else "0L"

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]
  if max_buffers is not None and len(bufs) > max_buffers:
    raise Unsupported(f"it uses {len(bufs)} arrays ({target} kernels are limited to {max_buffers})")

  for t in _plan_types(plan):
    em.ctype(t)
  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)

  if not is_cuda:
    scalars = sorted(scalars, key=lambda v: -v.type.itemsize)

  if is_cuda:
    hdr = (
      "#ifndef PK_CUDA_SIMULATE\n"
      "#include <mma.h>\n"
      "__device__ __forceinline__ void pk_cp_async_16(void* smem_ptr, const void* gmem_ptr) {\n"
      "  unsigned int s;\n"
      '  asm("{\\n .reg .u64 smem64;\\n cvta.to.shared.u64 smem64, %1;\\n cvt.u32.u64 %0, smem64;\\n}\\n" : "=r"(s) : "l"(smem_ptr));\n'
      '  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(gmem_ptr));\n'
      "}\n"
      "__device__ __forceinline__ void pk_cp_async_commit() {\n"
      '  asm volatile("cp.async.commit_group;\\n" ::);\n'
      "}\n"
      "__device__ __forceinline__ void pk_cp_async_wait() {\n"
      '  asm volatile("cp.async.wait_group 0;\\n" ::);\n'
      "}\n"
      "#else\n"
      "static inline void pk_cp_async_16(void* smem_ptr, const void* gmem_ptr) {\n"
      "  std::memcpy(smem_ptr, gmem_ptr, 16);\n"
      "}\n"
      "static inline void pk_cp_async_commit() {}\n"
      "static inline void pk_cp_async_wait() {}\n"
      "#endif\n" + prelude
    )
  else:
    hdr = prelude.replace("#include <metal_stdlib>", "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>", 1)

  lines: list[str] = [hdr]
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
    params.append("uint pk_sgid_u [[simdgroup_index_in_threadgroup]]")
    params.append("uint pk_lane_u [[thread_index_in_simdgroup]]")
    lines.append("kernel void pk_kernel(" + ",\n    ".join(params) + ") {")
    for v in scalars:
      lines.append(f"  const {em.ctype(v.type)} {v.name} = pk_a.{v.name};")

  # Block / warp / SIMD-group coordinates (wm x wn warps, wn == 2)
  if is_cuda:
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})blockIdx.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})threadIdx.x;")
    lines.append(f"  const {idx_t} pk_sgid = pk_tid >> 5;")
    lines.append(f"  const {idx_t} pk_lane = pk_tid & 31;")
  else:
    lines.append(f"  const {idx_t} pk_blk = ({idx_t})pk_bid.x;")
    lines.append(f"  const {idx_t} pk_tid = ({idx_t})pk_ltid.x;")
    lines.append(f"  const {idx_t} pk_sgid = ({idx_t})pk_sgid_u;")
    lines.append(f"  const {idx_t} pk_lane = ({idx_t})pk_lane_u;")
  lines.append(f"  const {idx_t} pk_sg_m = pk_sgid >> 1;")
  lines.append(f"  const {idx_t} pk_sg_n = pk_sgid & 1;")

  pure_ep = _match_pure_store_epilogue(plan, fold) if (is_cuda and init_val == 0.0) else None
  split_k_out_buf = -1
  if pure_ep is not None:
    out_uid = pure_ep[1].buf.uid
    for b_i, bv in enumerate(bufs):
      if bv.uid == out_uid:
        split_k_out_buf = b_i
        break
  split_k_enabled = split_k_out_buf >= 0

  # Shape dimensions & block unraveling (supports arbitrary batch rank >= 0 and Split-K on CUDA)
  lines.append(f"  const {idx_t} pk_M = ({idx_t})({em.expr(plan.shape[-2])});")
  lines.append(f"  const {idx_t} pk_N = ({idx_t})({em.expr(plan.shape[-1])});")
  lines.append(f"  const {idx_t} pk_K = ({idx_t})({em.expr(plan.reduce.shape[0])});")
  lines.append(f"  const {idx_t} pk_nb_n = (pk_N + {bn - 1}) / {bn};")
  lines.append(f"  const {idx_t} pk_nb_m = (pk_M + {bm - 1}) / {bm};")
  if split_k_enabled:
    lines.append(f"  const {idx_t} pk_split_k = pk_csize > 1 ? pk_csize : 1;")
    lines.append(f"  const {idx_t} pk_sk = pk_blk % pk_split_k;")
    lines.append(f"  const {idx_t} pk_tile_blk = pk_blk / pk_split_k;")
    blk_expr = "pk_tile_blk"
  else:
    blk_expr = "pk_blk"
  lines.append(f"  const {idx_t} pk_blk_n = {blk_expr} % pk_nb_n;")
  lines.append(f"  const {idx_t} pk_rem0 = {blk_expr} / pk_nb_n;")
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

  # Shared / threadgroup tiles (pk_sh_a[0] also reused for zero-overhead epilogue staging)
  if is_cuda:
    lines.append(f"  __align__(16) __shared__ float pk_sh_a[2][{a_size}];")
    lines.append(f"  __align__(16) __shared__ float pk_sh_b[2][{b_size}];")
  else:
    lines.append(f"  {sh_qual} float pk_sh_a[{a_size}];")
    lines.append(f"  {sh_qual} float pk_sh_b[{b_size}];")

  init_lit = repr(float(init_val)) + ("f" if is_cuda else "")
  if is_cuda:
    lines.append("  using namespace nvcuda::wmma;")
    lines.append(f"  fragment<accumulator, 16, 16, 8, float> pk_c[{r_m}][{r_n}];")
    lines.append("  #pragma unroll")
    lines.append(f"  for (int pk_rm = 0; pk_rm < {r_m}; ++pk_rm) {{")
    lines.append("    #pragma unroll")
    lines.append(f"    for (int pk_rn = 0; pk_rn < {r_n}; ++pk_rn) {{")
    lines.append(f"      fill_fragment(pk_c[pk_rm][pk_rn], {init_lit});")
    lines.append("    }")
    lines.append("  }")
  else:
    lines.append("  simdgroup_float8x8 pk_c[4][4];")
    lines.append("  #pragma unroll")
    lines.append("  for (int pk_rm = 0; pk_rm < 4; ++pk_rm) {")
    lines.append("    #pragma unroll")
    lines.append("    for (int pk_rn = 0; pk_rn < 4; ++pk_rn) {")
    lines.append(f"      pk_c[pk_rm][pk_rn] = make_filled_simdgroup_matrix<float, 8, 8>({init_lit});")
    lines.append("    }")
    lines.append("  }")

  if is_cuda:
    if split_k_enabled:
      lines.append(f"  const {idx_t} pk_ktiles = (pk_K + {bk - 1}) / {bk};")
      lines.append(f"  const {idx_t} pk_kt_per = (pk_ktiles + pk_split_k - 1) / pk_split_k;")
      lines.append(f"  const {idx_t} pk_k_start = (pk_sk * pk_kt_per) * {bk};")
      lines.append(f"  const {idx_t} pk_k_end = min(pk_K, ((pk_sk + 1) * pk_kt_per) * {bk});")
      lines.append("  if (pk_split_k > 1 && pk_k_start >= pk_K) return;")
    else:
      lines.append(f"  const {idx_t} pk_k_start = 0;")
      lines.append(f"  const {idx_t} pk_k_end = pk_K;")

    a_layout = "col_major" if a_col_major else "row_major"
    b_layout = "col_major" if b_col_major else "row_major"
    a_slow_v, a_fast_v = (st_row.index[1], st_row.index[0]) if a_col_major else (st_row.index[0], st_row.index[1])
    b_slow_v, b_fast_v = (st_col.index[0], st_col.index[1]) if b_col_major else (st_col.index[1], st_col.index[0])
    a_slow_lim, a_fast_lim = ("pk_K", "pk_M") if a_col_major else ("pk_M", "pk_K")
    b_slow_lim, b_fast_lim = ("pk_N", "pk_K") if b_col_major else ("pk_K", "pk_N")

    lines.append("  if (pk_k_start < pk_k_end) {")
    # Prologue: prefetch first K tile into buffer 0
    lines.append("    {")
    _emit_cuda_wmma_stage_load(
      em,
      st_row,
      "pk_sh_a",
      "0",
      a_slow,
      a_fast,
      a_stride,
      a_slow_v,
      a_fast_v,
      "pk_k_start" if a_col_major else "pk_i0",
      "pk_i0" if a_col_major else "pk_k_start",
      a_slow_lim,
      a_fast_lim,
      threads,
      a_unit,
      lines,
      "      ",
    )
    _emit_cuda_wmma_stage_load(
      em,
      st_col,
      "pk_sh_b",
      "0",
      b_slow,
      b_fast,
      b_stride,
      b_slow_v,
      b_fast_v,
      "pk_j0" if b_col_major else "pk_k_start",
      "pk_k_start" if b_col_major else "pk_j0",
      b_slow_lim,
      b_fast_lim,
      threads,
      b_unit,
      lines,
      "      ",
    )
    lines.append("      pk_cp_async_commit();")
    lines.append("      pk_cp_async_wait();")
    lines.append(f"      {barrier}")
    lines.append("    }")

    # Software-pipelined K-loop: 1 __syncthreads() per BK step, async load overlapped with WMMA!
    lines.append("    int pk_stage = 0;")
    lines.append(f"    for ({idx_t} pk_k0 = pk_k_start; pk_k0 < pk_k_end; pk_k0 += {bk}) {{")
    lines.append(f"      const {idx_t} pk_k_next = pk_k0 + {bk};")
    lines.append("      const int pk_next = pk_stage ^ 1;")
    lines.append("      if (pk_k_next < pk_k_end) {")
    _emit_cuda_wmma_stage_load(
      em,
      st_row,
      "pk_sh_a",
      "pk_next",
      a_slow,
      a_fast,
      a_stride,
      a_slow_v,
      a_fast_v,
      "pk_k_next" if a_col_major else "pk_i0",
      "pk_i0" if a_col_major else "pk_k_next",
      a_slow_lim,
      a_fast_lim,
      threads,
      a_unit,
      lines,
      "        ",
    )
    _emit_cuda_wmma_stage_load(
      em,
      st_col,
      "pk_sh_b",
      "pk_next",
      b_slow,
      b_fast,
      b_stride,
      b_slow_v,
      b_fast_v,
      "pk_j0" if b_col_major else "pk_k_next",
      "pk_k_next" if b_col_major else "pk_j0",
      b_slow_lim,
      b_fast_lim,
      threads,
      b_unit,
      lines,
      "        ",
    )
    lines.append("        pk_cp_async_commit();")
    lines.append("      }")

    lines.append("      #pragma unroll")
    lines.append(f"      for ({idx_t} pk_tk = 0; pk_tk < {bk}; pk_tk += 8) {{")
    lines.append(f"        fragment<matrix_a, 16, 16, 8, precision::tf32, {a_layout}> pk_fa[{r_m}];")
    lines.append(f"        fragment<matrix_b, 16, 16, 8, precision::tf32, {b_layout}> pk_fb[{r_n}];")
    for rm in range(r_m):
      if a_col_major:
        a_off = f"pk_tk * {a_stride} + pk_sg_m * {warp_m} + {rm * 16}"
      else:
        a_off = f"(pk_sg_m * {warp_m} + {rm * 16}) * {a_stride} + pk_tk"
      lines.append(f"        load_matrix_sync(pk_fa[{rm}], &pk_sh_a[pk_stage][{a_off}], {a_stride});")
    for rn in range(r_n):
      if b_col_major:
        b_off = f"(pk_sg_n * {warp_n} + {rn * 16}) * {b_stride} + pk_tk"
      else:
        b_off = f"pk_tk * {b_stride} + pk_sg_n * {warp_n} + {rn * 16}"
      lines.append(f"        load_matrix_sync(pk_fb[{rn}], &pk_sh_b[pk_stage][{b_off}], {b_stride});")
    for rm in range(r_m):
      for rn in range(r_n):
        lines.append(f"        mma_sync(pk_c[{rm}][{rn}], pk_fa[{rm}], pk_fb[{rn}], pk_c[{rm}][{rn}]);")
    lines.append("      }")
    lines.append("      if (pk_k_next < pk_k_end) {")
    lines.append("        pk_cp_async_wait();")
    lines.append(f"        {barrier}")
    lines.append("      }")
    lines.append("      pk_stage = pk_next;")
    lines.append("    }")
    lines.append(f"    {barrier}")
    lines.append("  }")
  else:
    bm_shift = bm.bit_length() - 1
    bn_shift = bn.bit_length() - 1
    lines.append(f"  for ({idx_t} pk_k0 = 0; pk_k0 < pk_K; pk_k0 += {bk}) {{")
    # Cooperative load of row stage A[BM x BK] (coalesced along whichever of i/k is unit-stride)
    lines.append(f"    for ({idx_t} pk_idx = pk_tid; pk_idx < {bm * bk}; pk_idx += {threads}) {{")
    if row_d0_fast:
      lines.append(f"      const {idx_t} pk_tk = pk_idx >> {bm_shift};")
      lines.append(f"      const {idx_t} pk_ti = pk_idx & {bm - 1};")
    else:
      lines.append(f"      const {idx_t} pk_ti = pk_idx >> {bk_shift};")
      lines.append(f"      const {idx_t} pk_tk = pk_idx & {bk - 1};")
    lines.append(f"      const {idx_t} {st_row.index[0].name} = pk_i0 + pk_ti;")
    lines.append(f"      const {idx_t} {st_row.index[1].name} = pk_k0 + pk_tk;")
    lines.append(f"      if ({st_row.index[0].name} < pk_M && {st_row.index[1].name} < pk_K) {{")
    _emit_decl_locals(em, st_row.body, {st_row.index[0].uid, st_row.index[1].uid}, lines, "        ")
    em.block(st_row.body, lines, 4)
    lines.append(f"        pk_sh_a[pk_ti * {a_stride} + pk_tk] = (float)({em.expr(st_row.value)});")
    lines.append("      } else {")
    lines.append(f"        pk_sh_a[pk_ti * {a_stride} + pk_tk] = 0.0f;")
    lines.append("      }")
    lines.append("    }")
    # Cooperative load of col stage B (coalesced along whichever of j/k is unit-stride)
    lines.append(f"    for ({idx_t} pk_idx = pk_tid; pk_idx < {bn * bk}; pk_idx += {threads}) {{")
    if col_d0_fast:
      lines.append(f"      const {idx_t} pk_tk = pk_idx >> {bn_shift};")
      lines.append(f"      const {idx_t} pk_tj = pk_idx & {bn - 1};")
    else:
      lines.append(f"      const {idx_t} pk_tj = pk_idx >> {bk_shift};")
      lines.append(f"      const {idx_t} pk_tk = pk_idx & {bk - 1};")
    lines.append(f"      const {idx_t} {st_col.index[0].name} = pk_j0 + pk_tj;")
    lines.append(f"      const {idx_t} {st_col.index[1].name} = pk_k0 + pk_tk;")
    lines.append(f"      if ({st_col.index[0].name} < pk_N && {st_col.index[1].name} < pk_K) {{")
    _emit_decl_locals(em, st_col.body, {st_col.index[0].uid, st_col.index[1].uid}, lines, "        ")
    em.block(st_col.body, lines, 4)
    lines.append(f"        pk_sh_b[pk_tj * {b_stride} + pk_tk] = (float)({em.expr(st_col.value)});")
    lines.append("      } else {")
    lines.append(f"        pk_sh_b[pk_tj * {b_stride} + pk_tk] = 0.0f;")
    lines.append("      }")
    lines.append("    }")
    lines.append(f"    {barrier}")
    lines.append("    #pragma unroll")
    lines.append(f"    for ({idx_t} pk_tk = 0; pk_tk < {bk}; pk_tk += 8) {{")
    lines.append("      simdgroup_float8x8 pk_fa[4];")
    lines.append("      simdgroup_float8x8 pk_fb[4];")
    for rm in range(4):
      lines.append(
        f"      simdgroup_load(pk_fa[{rm}], &pk_sh_a[(pk_sg_m * 32 + {rm * 8}) * {a_stride} + pk_tk], {a_stride});"
      )
    for rn in range(4):
      lines.append(
        f"      simdgroup_load(pk_fb[{rn}], &pk_sh_b[(pk_sg_n * 32 + {rn * 8}) * {b_stride} + pk_tk], {b_stride}, ulong2(0, 0), true);"
      )
    for rm in range(4):
      for rn in range(4):
        lines.append(
          f"      simdgroup_multiply_accumulate(pk_c[{rm}][{rn}], pk_fa[{rm}], pk_fb[{rn}], pk_c[{rm}][{rn}]);"
        )
    lines.append("    }")
    lines.append(f"    {barrier}")
    lines.append("  }")

  # Epilogue: stage tiles through dead pk_sh_a[0] and run plan.after
  epilogue_before = _collect_epilogue_before(plan.before, plan.after)
  epilogue_stmts = epilogue_before + plan.after
  i_var, j_var = plan.index[-2], plan.index[-1]
  skip_ep = {fold.result.uid, i_var.uid, j_var.uid}

  if is_cuda:
    lines.append("  float* pk_ep = &pk_sh_a[0][pk_sgid * 256];")
    lines.append(f"  for (int pk_rm = 0; pk_rm < {r_m}; ++pk_rm) {{")
    lines.append(f"    for (int pk_rn = 0; pk_rn < {r_n}; ++pk_rn) {{")
    lines.append("      store_matrix_sync(pk_ep, pk_c[pk_rm][pk_rn], 16, mem_row_major);")
    lines.append("      __syncwarp();")
    lines.append(f"      for ({idx_t} pk_e = pk_lane; pk_e < 256; pk_e += 32) {{")
    lines.append(f"        const {idx_t} {i_var.name} = pk_i0 + pk_sg_m * {warp_m} + pk_rm * 16 + (pk_e >> 4);")
    lines.append(f"        const {idx_t} {j_var.name} = pk_j0 + pk_sg_n * {warp_n} + pk_rn * 16 + (pk_e & 15);")
    lines.append(f"        if ({i_var.name} < pk_M && {j_var.name} < pk_N) {{")
    lines.append(f"          const float {fold.result.name} = pk_ep[pk_e];")
    if split_k_enabled and pure_ep is not None:
      prefix_ep, store_ep = pure_ep
      _emit_decl_locals(em, epilogue_stmts, skip_ep, lines, "          ")
      em.block(prefix_ep, lines, 5)
      st_idx = em.expr(store_ep.index)
      st_val = em.expr(store_ep.value)
      lines.append("          if (pk_split_k > 1) {")
      lines.append(f"            atomicAdd(&{em.data(store_ep.buf)}[{st_idx}], {st_val});")
      lines.append("          } else {")
      lines.append(f"            {em.data(store_ep.buf)}[{st_idx}] = {st_val};")
      lines.append("          }")
    else:
      _emit_decl_locals(em, epilogue_stmts, skip_ep, lines, "          ")
      em.block(epilogue_stmts, lines, 5)
    lines.append("        }")
    lines.append("      }")
    lines.append("      __syncwarp();")
    lines.append("    }")
    lines.append("  }")
  else:
    lines.append("  threadgroup float* pk_ep = &pk_sh_a[pk_sgid * 256];")
    lines.append("  for (int pk_rm = 0; pk_rm < 4; ++pk_rm) {")
    for rn in range(4):
      lines.append(f"    simdgroup_store(pk_c[pk_rm][{rn}], pk_ep + {rn * 8}, 32);")
    lines.append("    simdgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append(f"    for ({idx_t} pk_e = pk_lane; pk_e < 256; pk_e += 32) {{")
    lines.append(f"      const {idx_t} {i_var.name} = pk_i0 + pk_sg_m * 32 + pk_rm * 8 + (pk_e >> 5);")
    lines.append(f"      const {idx_t} {j_var.name} = pk_j0 + pk_sg_n * 32 + (pk_e & 31);")
    lines.append(f"      if ({i_var.name} < pk_M && {j_var.name} < pk_N) {{")
    lines.append(f"        const float {fold.result.name} = pk_ep[pk_e];")
    _emit_decl_locals(em, epilogue_stmts, skip_ep, lines, "        ")
    em.block(epilogue_stmts, lines, 4)
    lines.append("      }")
    lines.append("    }")
    lines.append("    simdgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append("  }")

  lines.append("}")

  written = {s.buf.uid for s in ir.walk_stmts(plan.after) if isinstance(s, ir.Store)}
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in plan.shape]
  tiled_info: dict = {
    "bm": bm,
    "bn": bn,
    "bk": bk,
    "tm": 4,
    "tn": 4,
    "threads": threads,
    "matrix_unit": "wmma" if is_cuda else "simdgroup_matrix",
    "shape": shape_meta,
  }
  if split_k_enabled:
    kd = plan.reduce.shape[0]
    tiled_info["split_k"] = {
      "k_dim": ("c", int(kd.value)) if isinstance(kd, ir.Const) else ("v", kd.name),
      "out_buf_idx": split_k_out_buf,
    }
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={"tiled": tiled_info},
  )


def generate_warp_reduce_cuda_kernel(
  loop: ir.ParLoop,
  free: list[ir.Var],
  em: CFamilyEmitter,
  prelude: str,
) -> KernelPlan | None:
  """Generate a CUDA 1-warp-per-row coalesced reduction kernel for
  ``loop.warp_reduce`` (8 warps = 256 threads/block, 100% coalesced stride-32
  reads, 5-step ``__shfl_down_sync`` reduction with zero shared memory)."""
  plan = loop.warp_reduce
  if plan is None:
    return None
  red = plan.reduce
  folds = red.folds
  if len(red.shape) != 1 or not folds:
    return None

  scalars = [v for v in free if not isinstance(v.type, BufType)]
  bufs = [v for v in free if isinstance(v.type, BufType)]

  for v in free:
    em.ctype(v.type.elt if isinstance(v.type, BufType) else v.type)
  for e in (*plan.shape, *red.shape):
    for x in ir.walk_expr(e):
      if isinstance(x.type, ScalarType):
        em.ctype(x.type)
  for f in folds:
    em.ctype(f.a.type)

  shfl_hdr = (
    "#ifndef PK_CUDA_SIMULATE\n"
    "template <typename T>\n"
    "__device__ __forceinline__ T pk_shfl_down(T v, int delta) {\n"
    "  return __shfl_down_sync(0xffffffffu, v, delta);\n"
    "}\n"
    "#else\n"
    "template <typename T>\n"
    "static inline T pk_shfl_down(T v, int delta) {\n"
    "  static T pk_shfl_buf[8][32];\n"
    "  int w = (int)((threadIdx.x >> 5) & 7u);\n"
    "  int l = (int)(threadIdx.x & 31u);\n"
    "  pk_shfl_buf[w][l] = v;\n"
    "  __syncwarp();\n"
    "  T res = (l + delta < 32) ? pk_shfl_buf[w][l + delta] : v;\n"
    "  __syncwarp();\n"
    "  return res;\n"
    "}\n"
    "#endif\n"
  )
  lines: list[str] = [shfl_hdr + prelude]
  params = ["long long pk_total", "long long pk_csize"]
  params += [f"{em.ctype(v.type)} {v.name}" for v in scalars]
  params += ["int* pk_flag", "long long* pk_info"]
  params += [f"{em.ctype(v.type.elt)}* {v.name}_d" for v in bufs]
  lines.append('extern "C" __global__ void pk_kernel(' + ",\n    ".join(params) + ") {")
  lines.append("  const long long pk_blk = (long long)blockIdx.x;")
  lines.append("  const long long pk_tid = (long long)threadIdx.x;")
  lines.append("  const long long pk_wid = pk_tid >> 5;")
  lines.append("  const int pk_lane = (int)(pk_tid & 31LL);")

  lines.append(f"  const long long pk_N = (long long)({em.expr(plan.shape[-1])});")
  lines.append(f"  const long long pk_K = (long long)({em.expr(red.shape[0])});")
  lines.append("  const long long pk_nb_n = (pk_N + 7LL) / 8LL;")
  lines.append("  const long long pk_blk_n = pk_blk % pk_nb_n;")
  lines.append(f"  const long long {plan.index[-1].name} = pk_blk_n * 8LL + pk_wid;")
  lines.append(f"  if ({plan.index[-1].name} >= pk_N) return;")

  if len(plan.shape) >= 2:
    lines.append("  long long pk_rem = pk_blk / pk_nb_n;")
    for ax in range(len(plan.shape) - 2, 0, -1):
      d_e = em.expr(plan.shape[ax])
      iv = plan.index[ax]
      lines.append(f"  const long long {iv.name} = pk_rem % (long long)({d_e});")
      lines.append(f"  pk_rem = pk_rem / (long long)({d_e});")
    lines.append(f"  const long long {plan.index[0].name} = pk_rem;")

  # Empty-reduction check for unseeded folds (e.g. np.max on 0-length axis)
  empty_errs = [f.empty for f in folds if f.empty is not None]
  if empty_errs:
    eid = em.error_id(empty_errs[0])
    lines.append("  if (pk_K <= 0LL) {")
    lines.append(f"    if (pk_lane == 0) pk_fail(pk_flag, pk_info, {eid}, 0LL, 0LL, 0LL, 0LL);")
    lines.append("    return;")
    lines.append("  }")

  # Uniform row setup (plan.before)
  skip_before = {v.uid for v in plan.index}
  _emit_decl_locals(em, plan.before, skip_before, lines, "  ")
  em.block(plan.before, lines, 1)

  # Initialize per-lane fold accumulators
  has_unseeded = any(f.op in ("max", "min") for f in folds)
  acc_vars = [ir.Var(f"pk_acc_{fi}", f.a.type) for fi, f in enumerate(folds)]
  for fi, f in enumerate(folds):
    ct = em.ctype(f.a.type)
    av = acc_vars[fi]
    if f.op == "mul":
      lines.append(f"  {ct} {av.name} = ({ct})1;")
    else:
      lines.append(f"  {ct} {av.name} = ({ct})0;")
  if has_unseeded:
    lines.append("  int pk_has = 0;")

  # Coalesced stride-32 warp reduction loop
  k_var = red.index[0]
  lines.append(f"  for (long long {k_var.name} = (long long)pk_lane; {k_var.name} < pk_K; {k_var.name} += 32LL) {{")
  _emit_decl_locals(em, red.body, {k_var.uid}, lines, "    ")
  em.block(red.body, lines, 2)
  for fi, f in enumerate(folds):
    ct = em.ctype(f.a.type)
    av = acc_vars[fi]
    ve = em.expr(f.value)
    if f.op == "add":
      lines.append(f"    {av.name} += ({ct})({ve});")
    elif f.op == "mul":
      lines.append(f"    {av.name} *= ({ct})({ve});")
    else:
      v_tmp = ir.Var(f"pk_val_{fi}", f.a.type)
      lines.append(f"    const {ct} {v_tmp.name} = ({ct})({ve});")
      comb = em.expr(ir.Prim(f.op, (av, v_tmp), f.a.type))
      lines.append(f"    {av.name} = pk_has ? ({comb}) : {v_tmp.name};")
  if has_unseeded:
    lines.append("    pk_has = 1;")
  lines.append("  }")

  # 5-step butterfly warp shuffle reduction (zero shared memory, zero __syncthreads!)
  lines.append("  #pragma unroll")
  lines.append("  for (int pk_off = 16; pk_off > 0; pk_off >>= 1) {")
  if has_unseeded:
    lines.append("    const int pk_ohas = pk_shfl_down(pk_has, pk_off);")
  for fi, f in enumerate(folds):
    ct = em.ctype(f.a.type)
    av = acc_vars[fi]
    o_tmp = ir.Var(f"pk_o_{fi}", f.a.type)
    lines.append(f"    const {ct} {o_tmp.name} = pk_shfl_down({av.name}, pk_off);")
    if f.op == "add":
      lines.append(f"    {av.name} += {o_tmp.name};")
    elif f.op == "mul":
      lines.append(f"    {av.name} *= {o_tmp.name};")
    else:
      comb = em.expr(ir.Prim(f.op, (av, o_tmp), f.a.type))
      lines.append(f"    if (pk_ohas) {av.name} = pk_has ? ({comb}) : {o_tmp.name};")
  if has_unseeded:
    lines.append("    pk_has |= pk_ohas;")
  lines.append("  }")

  # Lane 0 applies any non-identity init and runs plan.after (epilogue + stores)
  lines.append("  if (pk_lane == 0) {")
  for fi, f in enumerate(folds):
    ct = em.ctype(f.a.type)
    av = acc_vars[fi]
    if f.init is not None:
      is_triv_id = isinstance(f.init, ir.Const) and (
        (f.op == "add" and f.init.value == 0) or (f.op == "mul" and f.init.value == 1)
      )
      if not is_triv_id:
        init_e = em.expr(f.init)
        if f.op == "add":
          lines.append(f"    {av.name} = ({ct})({init_e}) + {av.name};")
        elif f.op == "mul":
          lines.append(f"    {av.name} = ({ct})({init_e}) * {av.name};")
        else:
          i_tmp = ir.Var(f"pk_init_{fi}", f.a.type)
          lines.append(f"    const {ct} {i_tmp.name} = ({ct})({init_e});")
          comb = em.expr(ir.Prim(f.op, (i_tmp, av), f.a.type))
          lines.append(f"    {av.name} = pk_has ? ({comb}) : {i_tmp.name};")
    lines.append(f"    const {ct} {f.result.name} = {av.name};")
  skip_after = {v.uid for v in plan.index} | {f.result.uid for f in folds} | set(defined_vars(plan.before))
  _emit_decl_locals(em, plan.after, skip_after, lines, "    ")
  em.block(plan.after, lines, 2)
  lines.append("  }")
  lines.append("}")

  written = {s.buf.uid for s in ir.walk_stmts(plan.after) if isinstance(s, ir.Store)}
  raw_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in plan.shape]
  shape_meta = [("c", 1), raw_meta[0]] if len(raw_meta) == 1 else raw_meta
  tiled_info: dict = {
    "bm": 1,
    "bn": 8,
    "bk": 32,
    "tm": 1,
    "tn": 1,
    "threads": 256,
    "warp_reduce": True,
    "shape": shape_meta,
  }
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={"tiled": tiled_info},
  )


def generate_tiled_gpu_kernel(
  loop: ir.ParLoop,
  free: list[ir.Var],
  target: str,
  em: CFamilyEmitter,
  prelude: str,
  max_buffers: int | None = None,
  profile: TargetProfile | None = None,
) -> KernelPlan | None:
  """Generate a cooperative ``__shared__`` (``target="cuda"``) or ``threadgroup``
  (``target="metal"``) block-tiled GPU kernel for ``loop.shared_tile``.

  When ``profile`` reports a supported hardware matrix unit (Apple Silicon
  ``simdgroup_matrix`` for exact float32, or CUDA ``sm_80+`` ``wmma`` when
  ``profile.tensor_cores`` is enabled) and ``loop.shared_tile`` is a float32
  sum-of-products contraction, emits a matrix-unit kernel directly.

  Returns ``None`` if the plan exceeds SRAM limits (so the caller can fall back
  to the ordinary 1D GPU loop), or raises :class:`Unsupported` if the body uses
  types/ops unavailable on ``target`` (e.g. ``float64`` on Metal).
  """
  plan = loop.shared_tile
  if plan is None:
    return None

  if plan.row_fold is None and profile is not None and profile.matrix_unit is not None:
    mu = profile.matrix_unit
    use_mu = (target == "metal" and mu.kind == "simdgroup_matrix") or (
      target == "cuda" and mu.kind == "wmma" and profile.tensor_cores
    )
    if use_mu:
      mm = match_matrix_unit_plan(plan)
      if mm is not None:
        row_idx, col_idx, fold, init_val = mm
        return _generate_matrix_unit_kernel(
          loop, plan, free, target, em, prelude, row_idx, col_idx, fold, init_val, max_buffers=max_buffers
        )

  row_fold = plan.row_fold
  cfg = row_fold.tile_config if row_fold is not None else choose_tile_config(plan, profile=profile)
  if cfg is None:
    return None
  bm, bn, bk, tm, tn = cfg
  bk_shift = bk.bit_length() - 1
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

  if row_fold is not None:
    lines.append(f"  {sh_qual} {em.ctype(row_fold.scores.type.elt)} {em.data(row_fold.scores)}[{bm * bn}];")

  # Register accumulators for TM x TN micro-tile across all folds (and any carries)
  for f_idx, f in enumerate(plan.mixed_folds):
    ct = em.ctype(f.a.type)
    init_expr = em.expr(f.init) if f.init is not None else _zero_literal(f.a.type)
    for mi in range(tm):
      for nj in range(tn):
        lines.append(f"  {ct} pk_acc_{f_idx}_{mi}_{nj} = ({ct})({init_expr});")
        for c_idx, c in enumerate(f.carries):
          cct = em.ctype(c.a.type)
          lines.append(f"  {cct} pk_pacc_{f_idx}_{c_idx}_{mi}_{nj} = {_zero_literal(c.a.type)};")

  # Inspect which coordinate vars mixed_body / mixed_folds reference
  mixed_used = used_vars(plan.mixed_body)
  for f in plan.mixed_folds:
    for e in (f.value, *(c.value for c in f.carries)):
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
  bm_shift = bm.bit_length() - 1
  bn_shift = bn.bit_length() - 1
  for s_idx, (st, side) in enumerate(zip(plan.stages, plan.stage_sides)):
    ct = em.ctype(st.buf.type.elt)
    zero = _zero_literal(st.buf.type.elt)
    d0_fast = _stage_prefers_dim0_fast(st)
    if side == "row":
      total_elts = bm * bk
      lines.append(f"    for ({idx_t} pk_idx = pk_tid; pk_idx < {total_elts}; pk_idx += {BLOCK_THREADS}) {{")
      if d0_fast:
        lines.append(f"      const {idx_t} pk_tk = pk_idx >> {bm_shift};")
        lines.append(f"      const {idx_t} pk_ti = pk_idx & {bm - 1};")
      else:
        lines.append(f"      const {idx_t} pk_ti = pk_idx >> {bk_shift};")
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
      if d0_fast:
        lines.append(f"      const {idx_t} pk_tk = pk_idx >> {bn_shift};")
        lines.append(f"      const {idx_t} pk_tj = pk_idx & {bn - 1};")
      else:
        lines.append(f"      const {idx_t} pk_tj = pk_idx >> {bk_shift};")
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
        pacc_names = [f"pk_pacc_{f_idx}_{c_idx}_{mi}_{nj}" for c_idx in range(len(f.carries))]
        val_expr = em.expr(f.value)
        pval_exprs = [em.expr(c.value) for c in f.carries]
        if f.init is None:
          lines.append(f"{pad}if (pk_k0 == 0 && pk_tk == 0) {{")
          lines.append(f"{pad}  {acc_name} = ({em.ctype(f.a.type)})({val_expr});")
          for c, pacc, pval in zip(f.carries, pacc_names, pval_exprs):
            lines.append(f"{pad}  {pacc} = ({em.ctype(c.a.type)})({pval});")
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
    c.result.uid for f in plan.mixed_folds for c in f.carries
  }
  for mi in range(tm):
    for nj in range(tn):
      lines.append("  {")
      lines.append(f"    const {idx_t} {i_var.name} = pk_i0 + pk_ty + {mi * NTY};")
      lines.append(f"    const {idx_t} {j_var.name} = pk_j0 + pk_tx + {nj * NTX};")
      lines.append(f"    if ({i_var.name} < pk_M && {j_var.name} < pk_N) {{")
      for f_idx, f in enumerate(plan.mixed_folds):
        lines.append(f"      const {em.ctype(f.result.type)} {f.result.name} = pk_acc_{f_idx}_{mi}_{nj};")
        for c_idx, c in enumerate(f.carries):
          lines.append(f"      const {em.ctype(c.result.type)} {c.result.name} = pk_pacc_{f_idx}_{c_idx}_{mi}_{nj};")
      _emit_decl_locals(em, epilogue_stmts, fold_result_uids | {i_var.uid, j_var.uid}, lines, "      ")
      em.block(epilogue_stmts, lines, 3)
      lines.append("    }")
      lines.append("  }")

  if row_fold is not None:
    # The score producer and row reduction share SRAM. Each row folds a
    # contiguous reference interval and writes its original structured state.
    lines.append(f"  {barrier}")
    lines.append(f"  if (pk_tid < {bm} && pk_i0 + pk_tid < pk_M) {{")
    r, c = row_fold.origin
    lines.append(f"    const {idx_t} {r.name} = pk_i0, {c.name} = pk_j0;")
    lines.append(f"    const {idx_t} {row_fold.rows.index[0].name} = pk_tid;")
    lines.append(f"    const {idx_t} {row_fold.width.name} = (pk_N - pk_j0 < {bn}) ? pk_N - pk_j0 : {bn};")
    _emit_decl_locals(em, row_fold.rows.body, set(), lines, "    ")
    em.block(row_fold.rows.body, lines, 2)
    lines.append("  }")
  lines.append("}")

  stores = row_fold.rows.body if row_fold is not None else plan.after
  written = {s.buf.uid for s in ir.walk_stmts(stores) if isinstance(s, ir.Store)}
  shape_meta = [("c", int(d.value)) if isinstance(d, ir.Const) else ("v", d.name) for d in plan.shape]
  return KernelPlan(
    loop=loop,
    source="\n".join(lines),
    scalars=scalars,
    bufs=bufs,
    errors=em.errors,
    written={v.uid for v in bufs if v.uid in written},
    extra={"tiled": {"bm": bm, "bn": bn, "bk": bk, "tm": tm, "tn": tn, "threads": BLOCK_THREADS, "shape": shape_meta}},
  )


__all__ = [
  "BK",
  "BLOCK_THREADS",
  "DEFAULT_SHARED_BUDGET",
  "MATRIX_UNIT_THREADS",
  "choose_tile_config",
  "generate_tiled_gpu_kernel",
  "match_matrix_unit_plan",
  "tiled_num_blocks",
]
