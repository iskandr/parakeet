"""Law 4: Chunkwise State-Space Duality & Parallel Affine Prefix Scan (``TileScan``).

Transforms sequential recurrences (``ForRange`` over ``t = 0 .. T - 1``) into
parallel chunkwise forms with ``O(T / C)`` sequential depth and Tensor-Core
friendly dense batched GEMMs:

1. **Affine Linear Recurrences** (EMA / IIR filters, scalar Kalman filters,
   Mamba-1 / RG-LRU / Griffin, GBDT histogram prefix-sum split finding)::

       for t in range(T):
           s = a(t) * s + b(t)
           out[idx(t)] = s

   Exploits the associative affine composition monoid::

       (a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2),   e = (1, 0)

   Rewrites into a 3-stage parallel chunkwise scan (chunk size ``C = 64``):
   - Stage 1 (parallel ``IndexMap`` over ``T / C`` chunks): compute composite
     chunk operators ``(chunk_A[c], chunk_B[c])``.
   - Stage 2 (short sequential scan over ``T / C`` chunks — ``C x`` fewer steps):
     compute incoming state ``prefix_in[c]`` and final state ``s``.
   - Stage 3 (parallel ``IndexMap`` over ``T / C`` chunks): sweep each chunk
     independently from ``prefix_in[c]`` and emit ``out[idx(t)]``.

2. **Bilinear Matrix-State Recurrences — Mamba-2 SSD / Gated Linear Attention**::

       for t in range(T):
           S = a(t) * S + k(t)[:, None] * v(t)[None, :]
           out[t] = q(t) @ S

   Exploits Chunkwise State-Space Duality (semiseparable matrix decomposition)
   with chunk size ``C = 64`` and division-free cumulative decay products:
   - Step 0 (parallel): compute chunk decay products ``cum_a``, ``tail_a``,
     ``chunk_decay``, lower-triangular ``L[c, i, j] = prod_{s=j+1..i} a[c,s]``,
     and zero-padded contiguous chunk tiles ``Q_c, K_c, V_c, K_tail``.
   - Step 1 (batched GEMMs -> Tensor Cores / register tiles):
     ``M[c,i,j] = L[c,i,j] * (Q_c[c,i] . K_c[c,j])``, ``Y_intra = M @ V_c``.
   - Step 2 (batched GEMM -> Tensor Cores / register tiles):
     ``B_chunk[c] = K_tail[c]^T @ V_c[c]``.
   - Step 3 (``C x`` shorter inter-chunk scan):
     ``S_in[c+1] = chunk_decay[c] * S_in[c] + B_chunk[c]``.
   - Step 4 (batched GEMM + parallel combine):
     ``Y_off = Q_c @ S_in``, ``out[t] = Y_intra[c,i] + cum_a[c,i] * Y_off[c,i]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ndtypes import BufType, Float32, Float64, ScalarType
from ..syntax import ir, ops
from ..syntax.clone import Cloner
from ..syntax.ir import (
  Alloc,
  Assign,
  Check,
  Const,
  Expr,
  Fold,
  ForRange,
  If,
  Incref,
  IndexMap,
  IndexReduce,
  Load,
  Prim,
  Release,
  ReleaseScope,
  Scope,
  Stmt,
  Store,
  Var,
)
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imin, imod, imul, iprod, isub
from .analysis import IndexFacts, is_pure_expr
from .lowering import row_major_index
from .monoid_tile import _expr_uses_uids, _resolve_local

AFFINE_CHUNK = 64
SSD_CHUNK = 64


def _is_unit_for_range(s: Stmt) -> bool:
  return (
    isinstance(s, ForRange)
    and isinstance(s.start, Const)
    and int(s.start.value) == 0
    and isinstance(s.step, Const)
    and int(s.step.value) == 1
  )


def _extract_affine_update(expr: Expr, state_uid: int, elt: ScalarType) -> tuple[Expr, Expr] | None:
  """Match ``a * s + b``, ``b + a * s``, ``s * a + b``, ``s + b``, ``b + s``, or ``a * s``.

  Returns ``(a_expr, b_expr)`` where neither depends on ``state_uid``, or ``None``.
  """
  if not isinstance(expr, Prim):
    return None
  one_c = Const(1.0, elt)
  zero_c = Const(0.0, elt)

  def _is_state(e: Expr) -> bool:
    return isinstance(e, Var) and e.uid == state_uid

  def _has_state(e: Expr) -> bool:
    return _expr_uses_uids(e, {state_uid})

  def _extract_mul(e: Expr) -> Expr | None:
    if _is_state(e):
      return one_c
    if isinstance(e, Prim) and e.op == "mul" and len(e.args) == 2:
      x, y = e.args
      if _is_state(x) and not _has_state(y):
        return y
      if _is_state(y) and not _has_state(x):
        return x
    return None

  if expr.op == "add" and len(expr.args) == 2:
    x, y = expr.args
    ax = _extract_mul(x)
    if ax is not None and not _has_state(y):
      return ax, y
    ay = _extract_mul(y)
    if ay is not None and not _has_state(x):
      return ay, x
    return None
  if expr.op == "sub" and len(expr.args) == 2:
    x, y = expr.args
    ax = _extract_mul(x)
    if ax is not None and not _has_state(y):
      return ax, ops.prim("neg", (y,), elt)
    return None
  if expr.op == "mul" and len(expr.args) == 2:
    a = _extract_mul(expr)
    if a is not None:
      return a, zero_c
  return None


def _hoist_upper_checks(checks: list[Check], t_var: Var, t_stop: Expr) -> list[Stmt]:
  """Hoist ``check(t < bound)`` out of ``for t in range(0, T, 1)`` as ``if T > 0: check(T - 1 < bound)``."""
  hoisted: list[Stmt] = []
  for chk in checks:
    c = chk.cond
    if (
      isinstance(c, Prim)
      and c.op == "lt"
      and len(c.args) == 2
      and isinstance(c.args[0], Var)
      and c.args[0].uid == t_var.uid
      and not _expr_uses_uids(c.args[1], {t_var.uid})
    ):
      last_t = isub(t_stop, ONE)
      cl = Cloner({t_var.uid: last_t})
      hoisted.append(Check(cmp("lt", last_t, c.args[1]), chk.err, tuple(cl.expr(a) for a in chk.args)))
    elif not _expr_uses_uids(c, {t_var.uid}):
      hoisted.append(chk)
    else:
      return []  # Cannot safely hoist unknown check
  if not hoisted:
    return []
  return [If(cmp("gt", t_stop, ZERO), hoisted, [])]


# ---------------------------------------------------------------------------
# Case 1: Scalar / Elementwise Affine Prefix Scan (IIR / EMA / Kalman / LRU / GBDT)
# ---------------------------------------------------------------------------


@dataclass
class _AffineLoopMatch:
  state_var: Var
  elt: ScalarType
  a_expr: Expr
  b_expr: Expr
  pre_assigns: list[Assign]
  stores: list[tuple[Store, bool]]  # (store, before_update: True = exclusive, False = inclusive)
  checks: list[Check]
  next_var_uid: int


def _match_affine_for_range(loop: ForRange, outer_vars: dict[int, Var]) -> _AffineLoopMatch | None:
  if not _is_unit_for_range(loop):
    return None
  body = [s for s in loop.body if not isinstance(s, (ir.Comment, ReleaseScope))]
  if not body or any(not isinstance(s, (Assign, Store, Check)) for s in body):
    return None

  # Find loop-carried scalar state update: Assign(s_outer, val) where s_outer is defined outside loop
  carried: list[Assign] = [s for s in body if isinstance(s, Assign) and s.var.uid in outer_vars]
  if len(carried) != 1:
    return None
  update_stmt = carried[0]
  upd_pos = body.index(update_stmt)
  state_var = update_stmt.var
  elt = state_var.type
  if elt not in (Float32, Float64):
    return None

  # Map local assigns (excluding update_stmt)
  local_assigns = [s for s in body if isinstance(s, Assign) and s is not update_stmt]
  val_map = {s.var.uid: s.value for s in local_assigns}
  resolved_next = _resolve_local(update_stmt.value, val_map)
  ab = _extract_affine_update(resolved_next, state_var.uid, elt)
  if ab is None:
    return None
  a_expr, b_expr = ab
  if not (is_pure_expr(a_expr) and is_pure_expr(b_expr)):
    return None

  # Check stores and classify each as exclusive (pre-update state) vs inclusive (post-update state)
  raw_stores = [(idx_s, s) for idx_s, s in enumerate(body) if isinstance(s, Store)]
  if not raw_stores:
    return None
  next_var_uid = update_stmt.value.uid if isinstance(update_stmt.value, Var) else -1
  written_bufs = {st.buf.uid for _, st in raw_stores}
  # Ensure a_expr and b_expr do not read from any buffer written by stores (no memory-carried loop dep!)
  for ex in (a_expr, b_expr):
    for x in ir.walk_expr(ex):
      if isinstance(x, Load) and x.buf.uid in written_bufs:
        return None
  classified_stores: list[tuple[Store, bool]] = []
  for st_pos, st in raw_stores:
    if st.atomic is not None:
      return None
    if _expr_uses_uids(st.index, {state_var.uid, next_var_uid}):
      return None
    sv = st.value
    if isinstance(sv, Var) and sv.uid == state_var.uid:
      # Reading state_var before update_stmt is exclusive; after update_stmt is inclusive
      classified_stores.append((st, st_pos < upd_pos))
    elif (isinstance(sv, Var) and sv.uid == next_var_uid) or ir.same(_resolve_local(sv, val_map), resolved_next):
      # Explicitly storing the computed next state is always inclusive
      classified_stores.append((st, False))
    else:
      return None

  checks = [s for s in body if isinstance(s, Check)]
  # Ensure all checks can be hoisted safely
  if checks and not _hoist_upper_checks(checks, loop.var, loop.stop):
    return None

  return _AffineLoopMatch(
    state_var=state_var,
    elt=elt,
    a_expr=a_expr,
    b_expr=b_expr,
    pre_assigns=local_assigns,
    stores=classified_stores,
    checks=checks,
    next_var_uid=next_var_uid,
  )


def _rewrite_affine_loop(
  loop: ForRange,
  m: _AffineLoopMatch,
  owner: Scope,
  batch_shape: tuple[Expr, ...] = (),
  batch_index: tuple[Var, ...] = (),
  init_expr: Expr | None = None,
) -> list[Stmt]:
  """Rewrite an affine recurrence into a 3-stage parallel chunkwise scan (``TileScan``)."""
  elt = m.elt
  t_stop = loop.stop
  t_var = loop.var
  chunk_c = Const(AFFINE_CHUNK, I)
  one_c = Const(1.0, elt)
  zero_c = Const(0.0, elt)

  hoisted_checks = _hoist_upper_checks(m.checks, t_var, t_stop)
  nc = Var("ts_nc", I)
  nc_assign = Assign(nc, idiv(iadd(t_stop, Const(AFFINE_CHUNK - 1, I)), chunk_c))

  chunk_shape = (*batch_shape, nc)
  chunk_count = iprod(chunk_shape)
  ch_a = Var("ts_chA", BufType(elt))
  ch_b = Var("ts_chB", BufType(elt))
  ch_in = Var("ts_in", BufType(elt))
  owner.owned.extend([ch_a, ch_b, ch_in])

  # Stage 1: Parallel IndexMap over (*batch, nc) computing composite (A_c, B_c)
  s1_b = tuple(Var("ts1b", I) for _ in batch_shape)
  s1_c = Var("ts1c", I)
  s1_t = Var("ts1t", I)
  s1_start = Var("ts1s", I)
  s1_stop = Var("ts1e", I)
  s1_aa = Var("ts1aa", elt)
  s1_bb = Var("ts1bb", elt)
  cl1 = Cloner({**{bv.uid: nb for bv, nb in zip(batch_index, s1_b)}, t_var.uid: s1_t})
  a1 = cl1.expr(m.a_expr)
  b1 = cl1.expr(m.b_expr)
  av1 = Var("ts1_av", elt)
  bv1 = Var("ts1_bv", elt)
  s1_loop_body: list[Stmt] = [
    Assign(av1, a1),
    Assign(bv1, b1),
    Assign(s1_aa, ops.prim("mul", (av1, s1_aa), elt)),
    Assign(s1_bb, ops.prim("add", (ops.prim("mul", (av1, s1_bb), elt), bv1), elt)),
  ]
  s1_pos = row_major_index((*s1_b, s1_c), chunk_shape)
  s1_scope = Scope("loop")
  s1_body: list[Stmt] = [
    Assign(s1_start, imul(s1_c, chunk_c)),
    Assign(s1_stop, imin(iadd(s1_start, chunk_c), t_stop)),
    Assign(s1_aa, one_c),
    Assign(s1_bb, zero_c),
    ForRange(s1_t, s1_start, s1_stop, ONE, s1_loop_body, Scope("loop")),
    Store(ch_a, s1_pos, s1_aa),
    Store(ch_b, s1_pos, s1_bb),
    ReleaseScope(s1_scope),
  ]
  stage1 = IndexMap(shape=chunk_shape, index=(*s1_b, s1_c), body=s1_body, scope=s1_scope, parallel=True, loc=None)

  # Stage 2: Inter-chunk prefix scan (C x shorter!)
  if not batch_shape:
    s2_c = Var("ts2c", I)
    s2_run = Var("ts2run", elt)
    s2_body: list[Stmt] = [
      Store(ch_in, s2_c, s2_run),
      Assign(
        s2_run,
        ops.prim("add", (ops.prim("mul", (Load(ch_a, s2_c), s2_run), elt), Load(ch_b, s2_c)), elt),
      ),
    ]
    stage2: list[Stmt] = [
      Assign(s2_run, m.state_var if init_expr is None else init_expr),
      ForRange(s2_c, ZERO, nc, ONE, s2_body, Scope("loop")),
      Assign(m.state_var, s2_run),
    ]
  else:
    s2_b = tuple(Var("ts2b", I) for _ in batch_shape)
    s2_c = Var("ts2c", I)
    s2_run = Var("ts2run", elt)
    s2_pos = row_major_index((*s2_b, s2_c), chunk_shape)
    cl2 = Cloner({bv.uid: nb for bv, nb in zip(batch_index, s2_b)})
    s2_init = cl2.expr(init_expr) if init_expr is not None else zero_c
    s2_loop = ForRange(
      s2_c,
      ZERO,
      nc,
      ONE,
      [
        Store(ch_in, s2_pos, s2_run),
        Assign(
          s2_run,
          ops.prim("add", (ops.prim("mul", (Load(ch_a, s2_pos), s2_run), elt), Load(ch_b, s2_pos)), elt),
        ),
      ],
      Scope("loop"),
    )
    s2_scope = Scope("loop")
    stage2 = [
      IndexMap(
        shape=batch_shape,
        index=s2_b,
        body=[Assign(s2_run, s2_init), s2_loop, ReleaseScope(s2_scope)],
        scope=s2_scope,
        parallel=True,
        loc=None,
      )
    ]

  # Stage 3: Parallel IndexMap over (*batch, nc) finalizing all outputs
  s3_b = tuple(Var("ts3b", I) for _ in batch_shape)
  s3_c = Var("ts3c", I)
  s3_t = Var("ts3t", I)
  s3_start = Var("ts3s", I)
  s3_stop = Var("ts3e", I)
  s3_st = Var("ts3st", elt)
  s3_pos = row_major_index((*s3_b, s3_c), chunk_shape)
  cl3 = Cloner({**{bv.uid: nb for bv, nb in zip(batch_index, s3_b)}, t_var.uid: s3_t})
  a3 = cl3.expr(m.a_expr)
  b3 = cl3.expr(m.b_expr)
  val_map = {a.var.uid: a.value for a in m.pre_assigns}
  s3_loop_body: list[Stmt] = []
  # Emit exclusive (pre-update) stores before advancing s3_st
  for st, before_update in m.stores:
    if before_update:
      st_idx_resolved = _resolve_local(st.index, val_map)
      s3_loop_body.append(Store(st.buf, cl3.expr(st_idx_resolved), cast(s3_st, st.buf.type.elt)))
  s3_loop_body.append(Assign(s3_st, ops.prim("add", (ops.prim("mul", (a3, s3_st), elt), b3), elt)))
  # Emit inclusive (post-update) stores after advancing s3_st
  for st, before_update in m.stores:
    if not before_update:
      st_idx_resolved = _resolve_local(st.index, val_map)
      s3_loop_body.append(Store(st.buf, cl3.expr(st_idx_resolved), cast(s3_st, st.buf.type.elt)))
  s3_scope = Scope("loop")
  s3_body: list[Stmt] = [
    Assign(s3_start, imul(s3_c, chunk_c)),
    Assign(s3_stop, imin(iadd(s3_start, chunk_c), t_stop)),
    Assign(s3_st, Load(ch_in, s3_pos)),
    ForRange(s3_t, s3_start, s3_stop, ONE, s3_loop_body, Scope("loop")),
    ReleaseScope(s3_scope),
  ]
  stage3 = IndexMap(shape=chunk_shape, index=(*s3_b, s3_c), body=s3_body, scope=s3_scope, parallel=True, loc=None)

  tiled_branch: list[Stmt] = [
    *hoisted_checks,
    nc_assign,
    Alloc(ch_a, chunk_count, zero=False),
    Alloc(ch_b, chunk_count, zero=False),
    Alloc(ch_in, chunk_count, zero=False),
    stage1,
    *stage2,
    stage3,
    Release(ch_a),
    Release(ch_b),
    Release(ch_in),
  ]

  if not batch_shape:
    return [If(cmp("gt", t_stop, chunk_c), tiled_branch, [loop])]
  return tiled_branch


# ---------------------------------------------------------------------------
# Case 2: Bilinear Matrix-State Recurrence (Mamba-2 SSD / Gated Linear Attention)
# ---------------------------------------------------------------------------


@dataclass
class _SSDMatch:
  t_var: Var
  t_dim: Expr
  d_dim: Expr
  v_dim: Expr
  elt: ScalarType
  state_buf_var: Var
  a_expr: Expr  # scalar decay a(t) (Const(1.0) if ungated)
  k_expr: Expr  # element k(t, d) using (t_var, d_var)
  v_expr: Expr  # element v(t, v) using (t_var, v_var)
  q_expr: Expr  # element q(t, d) using (t_var, d_var)
  d_var: Var
  v_var: Var
  out_buf: Var
  out_row_offset: Expr  # row base offset in out_buf for step t
  checks: list[Check]
  keep_var: Var | None


def _match_ssd_for_range(
  loop: ForRange,
  facts: IndexFacts,
  outer_vars: dict[int, Var],
) -> _SSDMatch | None:
  """Match Mamba-2 SSD / GLA bilinear matrix-state recurrence::

  S = a[t] * S + k[t, :, None] * v[t, None, :]
  out[t] = q[t] @ S
  """
  if not _is_unit_for_range(loop):
    return None
  t_var = loop.var
  t_dim = loop.stop
  body = loop.body

  # 1. Locate state-update 2D IndexMap: S_next[d, v] = a(t) * S[d, v] + k(t, d) * v(t, v)
  #    and verify S is reassigned to S_next at the end of the loop.
  maps = [(i, s) for i, s in enumerate(body) if isinstance(s, IndexMap)]
  if len(maps) < 2:
    return None

  outer_assigns = {s.var.uid: s.value for s in body if isinstance(s, Assign) and s.var.uid not in outer_vars}
  rebinds = [s for s in body if isinstance(s, Assign) and s.var.uid in outer_vars and isinstance(s.var.type, BufType)]
  if not rebinds:
    return None
  # Find the state buffer var (rebound to an Alloc inside the loop)
  loop_allocs = {s.var.uid: s for s in body if isinstance(s, Alloc)}
  state_buf_var: Var | None = None
  next_buf_var: Var | None = None
  keep_var: Var | None = None
  for rb in rebinds:
    if isinstance(rb.value, Var) and rb.value.uid in loop_allocs:
      # Distinguish S_25 (read in map0) from keep_75 (only released/reassigned)
      used_in_load = any(
        isinstance(x, Load) and x.buf.uid == rb.var.uid
        for _, m in maps
        for st in m.body
        if isinstance(st, (Assign, Store))
        for x in ir.walk_expr(st.value if isinstance(st, Assign) else st.value)
      )
      if used_in_load:
        state_buf_var = rb.var
        next_buf_var = rb.value
      else:
        keep_var = rb.var
  if state_buf_var is None or next_buf_var is None:
    return None
  elt = state_buf_var.type.elt
  if elt not in (Float32, Float64):
    return None

  # Find the 2D IndexMap that writes next_buf_var
  upd_idx: int | None = None
  upd_map: IndexMap | None = None
  for idx_m, m in maps:
    if len(m.shape) == 2 and any(isinstance(st, Store) and st.buf.uid == next_buf_var.uid for st in m.body):
      upd_idx = idx_m
      upd_map = m
      break
  if upd_map is None or upd_idx is None:
    return None

  stores_u = [st for st in upd_map.body if isinstance(st, Store)]
  if len(stores_u) != 1:
    return None
  st_u = stores_u[0]
  d_var, v_var = upd_map.index
  d_dim, v_dim = upd_map.shape
  # Resolve d_dim / v_dim through select(S_n == 1, k_n, S_n) using loop-invariant dims if possible
  # Wait: d_dim = select(S_n0 == 1, k_n1, S_n0), where S_n0 was initialized to k_n1!
  # Use outer facts to resolve!
  lf = facts.fork()
  for s in body[:upd_idx]:
    lf.record_stmt(s)

  # Parse st_u.value == a(t) * S[d, v] + k(t, d) * v(t, v)
  u_locals = {**outer_assigns, **{s.var.uid: s.value for s in upd_map.body if isinstance(s, Assign)}}
  rhs_u = _resolve_local(st_u.value, u_locals)
  if not (isinstance(rhs_u, Prim) and rhs_u.op == "add" and len(rhs_u.args) == 2):
    return None

  def _is_s_load(e: Expr) -> bool:
    return isinstance(e, Load) and e.buf.uid == state_buf_var.uid

  def _parse_decay_s(e: Expr) -> Expr | None:
    if _is_s_load(e):
      return Const(1.0, elt)
    if isinstance(e, Prim) and e.op == "mul" and len(e.args) == 2:
      if _is_s_load(e.args[1]) and not _expr_uses_uids(e.args[0], {d_var.uid, v_var.uid}):
        return e.args[0]
      if _is_s_load(e.args[0]) and not _expr_uses_uids(e.args[1], {d_var.uid, v_var.uid}):
        return e.args[1]
    return None

  def _parse_outer_kv(e: Expr) -> tuple[Expr, Expr] | None:
    if not (isinstance(e, Prim) and e.op == "mul" and len(e.args) == 2):
      return None
    x, y = e.args
    xd = _expr_uses_uids(x, {d_var.uid})
    xv = _expr_uses_uids(x, {v_var.uid})
    yd = _expr_uses_uids(y, {d_var.uid})
    yv = _expr_uses_uids(y, {v_var.uid})
    if xd and not xv and yv and not yd:
      return x, y
    if xv and not xd and yd and not yv:
      return y, x
    return None

  a_expr: Expr | None = None
  kv_pair: tuple[Expr, Expr] | None = None
  da = _parse_decay_s(rhs_u.args[0])
  if da is not None:
    a_expr = da
    kv_pair = _parse_outer_kv(rhs_u.args[1])
  else:
    da = _parse_decay_s(rhs_u.args[1])
    if da is not None:
      a_expr = da
      kv_pair = _parse_outer_kv(rhs_u.args[0])
  if a_expr is None or kv_pair is None:
    return None
  k_expr, v_expr = kv_pair

  # 2. Match readout: optional transpose `rows[v, d] = S_next[d, v]`, then contraction
  #    `tmp[v] = sum_d q(t, d) * S_next[d, v]`, then optional copy `out[t, v] = tmp[v]`
  trans_buf: Var | None = None
  read_map: IndexMap | None = None
  copy_map: IndexMap | None = None
  for idx_m, m in maps:
    if idx_m <= upd_idx:
      continue
    if len(m.shape) == 2 and trans_buf is None:
      # Check transpose of next_buf_var
      m_stores = [st for st in m.body if isinstance(st, Store)]
      if len(m_stores) == 1 and isinstance(m_stores[0].value, Load) and m_stores[0].value.buf.uid == next_buf_var.uid:
        trans_buf = m_stores[0].buf
        continue
    if len(m.shape) == 1 and read_map is None:
      if any(isinstance(st, IndexReduce) for st in m.body):
        read_map = m
        continue
    if len(m.shape) == 1 and read_map is not None and copy_map is None:
      copy_map = m
      continue
  if read_map is None:
    return None

  # Inspect read_map: 1D IndexMap over V with 1D IndexReduce over D summing q(t, d) * S_next[d, v]
  reds = [st for st in read_map.body if isinstance(st, IndexReduce)]
  r_stores = [st for st in read_map.body if isinstance(st, Store)]
  if len(reds) != 1 or len(r_stores) != 1:
    return None
  red = reds[0]
  if len(red.shape) != 1 or len(red.folds) != 1 or red.folds[0].op != "add":
    return None
  r_locals = {
    **outer_assigns,
    **{s.var.uid: s.value for s in read_map.body if isinstance(s, Assign)},
    **{s.var.uid: s.value for s in red.body if isinstance(s, Assign)},
  }
  rfv = _resolve_local(red.folds[0].value, r_locals)
  if not (isinstance(rfv, Prim) and rfv.op == "mul" and len(rfv.args) == 2):
    return None
  s_read_uids = {next_buf_var.uid}
  if trans_buf is not None:
    s_read_uids.add(trans_buf.uid)
  rk_var = red.index[0]
  if isinstance(rfv.args[0], Load) and rfv.args[0].buf.uid in s_read_uids:
    q_raw = rfv.args[1]
  elif isinstance(rfv.args[1], Load) and rfv.args[1].buf.uid in s_read_uids:
    q_raw = rfv.args[0]
  else:
    return None
  if not (_expr_uses_uids(q_raw, {rk_var.uid}) and not _expr_uses_uids(q_raw, {read_map.index[0].uid})):
    return None
  # Rebind q_raw's reduction index rk_var -> d_var
  q_expr = Cloner({rk_var.uid: d_var}).expr(q_raw)

  # Find output buffer and row offset
  read_out_buf = r_stores[0].buf
  if copy_map is not None:
    c_stores = [st for st in copy_map.body if isinstance(st, Store)]
    if len(c_stores) != 1:
      return None
    c_st = c_stores[0]
    if not (isinstance(c_st.value, Load) and c_st.value.buf.uid == read_out_buf.uid):
      return None
    out_buf = c_st.buf
    out_idx = _resolve_local(c_st.index, outer_assigns)
    c_v = copy_map.index[0]
    # Expect out_idx == off(t) + c_v
    if isinstance(out_idx, Prim) and out_idx.op == "add" and len(out_idx.args) == 2:
      if isinstance(out_idx.args[1], Var) and out_idx.args[1].uid == c_v.uid:
        out_row_offset = out_idx.args[0]
      elif isinstance(out_idx.args[0], Var) and out_idx.args[0].uid == c_v.uid:
        out_row_offset = out_idx.args[1]
      else:
        return None
    else:
      return None
  else:
    out_buf = read_out_buf
    r_v = read_map.index[0]
    out_idx = _resolve_local(r_stores[0].index, outer_assigns)
    if isinstance(out_idx, Prim) and out_idx.op == "add" and len(out_idx.args) == 2:
      if isinstance(out_idx.args[1], Var) and out_idx.args[1].uid == r_v.uid:
        out_row_offset = out_idx.args[0]
      elif isinstance(out_idx.args[0], Var) and out_idx.args[0].uid == r_v.uid:
        out_row_offset = out_idx.args[1]
      else:
        return None
    else:
      return None

  # Resolve loop-invariant D and V dimensions (strip broadcast select(S_n == 1, k_n, S_n) -> k_n)
  def _resolve_loop_inv_dim(d: Expr) -> Expr | None:
    dr = _resolve_local(d, outer_assigns)
    if isinstance(dr, Prim) and dr.op == "select" and len(dr.args) == 3:
      # select((S_n == 1), k_n, S_n) -> check if S_n is loop-invariant in facts
      _, t_e, f_e = dr.args
      if not _expr_uses_uids(t_e, set(outer_assigns)):
        return t_e
      if not _expr_uses_uids(f_e, set(outer_assigns)):
        return f_e
    if not _expr_uses_uids(dr, set(outer_assigns)):
      return dr
    return None

  d_inv = _resolve_loop_inv_dim(d_dim)
  v_inv = _resolve_loop_inv_dim(v_dim)
  if d_inv is None or v_inv is None:
    return None

  # Ensure a_expr, k_expr, v_expr, q_expr don't reference loop-local vars (except t_var, d_var, v_var)
  local_uids = set(outer_assigns) | set(loop_allocs)
  for ex in (a_expr, k_expr, v_expr, q_expr, out_row_offset):
    if _expr_uses_uids(ex, local_uids):
      return None
    if not is_pure_expr(ex):
      return None
    # No dependence on out_buf or state_buf_var
    for x in ir.walk_expr(ex):
      if isinstance(x, Load) and x.buf.uid in (out_buf.uid, state_buf_var.uid):
        return None

  checks = [s for s in body if isinstance(s, Check)]
  return _SSDMatch(
    t_var=t_var,
    t_dim=t_dim,
    d_dim=d_inv,
    v_dim=v_inv,
    elt=elt,
    state_buf_var=state_buf_var,
    a_expr=a_expr,
    k_expr=k_expr,
    v_expr=v_expr,
    q_expr=q_expr,
    d_var=d_var,
    v_var=v_var,
    out_buf=out_buf,
    out_row_offset=out_row_offset,
    checks=checks,
    keep_var=keep_var,
  )


def _batched_gemm_map(
  out_buf: Var,
  elt: ScalarType,
  nc: Expr,
  m_dim: Expr,
  n_dim: Expr,
  k_dim: Expr,
  lhs_buf: Var,
  lhs_trans: bool,
  rhs_buf: Var,
  rhs_trans: bool,
  mul_buf: Var | None = None,
  loc=None,
) -> IndexMap:
  """Emit a contiguous 3D batched GEMM ``IndexMap`` over ``(nc, M, N)`` contracted over ``K``."""
  c_v, i_v, j_v, k_v = Var("sc", I), Var("si", I), Var("sj", I), Var("sk", I)
  if not lhs_trans:
    # lhs: (nc, M, K)
    l_idx = row_major_index((c_v, i_v, k_v), (nc, m_dim, k_dim))
  else:
    # lhs: (nc, K, M)
    l_idx = row_major_index((c_v, k_v, i_v), (nc, k_dim, m_dim))
  if not rhs_trans:
    # rhs: (nc, K, N)
    r_idx = row_major_index((c_v, k_v, j_v), (nc, k_dim, n_dim))
  else:
    # rhs: (nc, N, K)
    r_idx = row_major_index((c_v, j_v, k_v), (nc, n_dim, k_dim))
  prod = Var("sp", elt)
  r_body: list[Stmt] = [Assign(prod, ops.prim("mul", (Load(lhs_buf, l_idx), Load(rhs_buf, r_idx)), elt))]
  a_v, b_v, res = Var("sa", elt), Var("sb", elt), Var("sred", elt)
  fold = Fold(
    result=res,
    value=prod,
    a=a_v,
    b=b_v,
    combine=[],
    combined=ops.prim("add", (a_v, b_v), elt),
    init=Const(0.0, elt),
    op="add",
  )
  scope = Scope("loop")
  red = IndexReduce(
    shape=(k_dim,),
    index=(k_v,),
    body=r_body,
    folds=[fold],
    scope=Scope("loop"),
    home=scope,
    parallel=False,
    loc=loc,
  )
  out_pos = row_major_index((c_v, i_v, j_v), (nc, m_dim, n_dim))
  val: Expr = res
  if mul_buf is not None:
    val = ops.prim("mul", (val, Load(mul_buf, out_pos)), elt)
  return IndexMap(
    shape=(nc, m_dim, n_dim),
    index=(c_v, i_v, j_v),
    body=[red, Store(out_buf, out_pos, val), ReleaseScope(scope)],
    scope=scope,
    parallel=True,
    loc=loc,
  )


def _rewrite_ssd_loop(loop: ForRange, m: _SSDMatch, owner: Scope) -> list[Stmt]:
  """Rewrite Mamba-2 SSD / GLA recurrence into 4-step Chunkwise State-Space Duality."""
  elt = m.elt
  t_dim, d_dim, v_dim = m.t_dim, m.d_dim, m.v_dim
  chunk_c = Const(SSD_CHUNK, I)
  zero_c = Const(0.0, elt)
  one_c = Const(1.0, elt)

  # Hoist bounds checks on t (skip broadcast checks on S_n0/S_n1 which use loop-local vars)
  t_checks = [
    chk
    for chk in m.checks
    if isinstance(chk.cond, Prim)
    and chk.cond.op == "lt"
    and len(chk.cond.args) == 2
    and isinstance(chk.cond.args[0], Var)
    and chk.cond.args[0].uid == m.t_var.uid
  ]
  hoisted_checks = _hoist_upper_checks(t_checks, m.t_var, t_dim)

  nc = Var("ssd_nc", I)
  nc_assign = Assign(nc, idiv(iadd(t_dim, Const(SSD_CHUNK - 1, I)), chunk_c))

  # Allocate chunk tables & padded contiguous tiles
  cum_a = Var("ssd_cuma", BufType(elt))  # (nc, C)
  tail_a = Var("ssd_taila", BufType(elt))  # (nc, C)
  ch_dec = Var("ssd_dec", BufType(elt))  # (nc,)
  l_mat = Var("ssd_L", BufType(elt))  # (nc, C, C)
  q_c = Var("ssd_Qc", BufType(elt))  # (nc, C, D)
  k_c = Var("ssd_Kc", BufType(elt))  # (nc, C, D)
  k_tail = Var("ssd_Kt", BufType(elt))  # (nc, C, D)
  v_c = Var("ssd_Vc", BufType(elt))  # (nc, C, V)
  m_mat = Var("ssd_M", BufType(elt))  # (nc, C, C)
  y_intra = Var("ssd_Yi", BufType(elt))  # (nc, C, V)
  b_ch = Var("ssd_B", BufType(elt))  # (nc, D, V)
  s_in = Var("ssd_Sin", BufType(elt))  # (nc + 1, D, V)
  y_off = Var("ssd_Yo", BufType(elt))  # (nc, C, V)
  all_tmps = [cum_a, tail_a, ch_dec, l_mat, q_c, k_c, k_tail, v_c, m_mat, y_intra, b_ch, s_in, y_off]
  owner.owned.extend(all_tmps)

  nc_c = imul(nc, chunk_c)
  allocs = [
    Alloc(cum_a, nc_c, zero=False),
    Alloc(tail_a, nc_c, zero=False),
    Alloc(ch_dec, nc, zero=False),
    Alloc(l_mat, imul(nc_c, chunk_c), zero=False),
    Alloc(q_c, imul(nc_c, d_dim), zero=False),
    Alloc(k_c, imul(nc_c, d_dim), zero=False),
    Alloc(k_tail, imul(nc_c, d_dim), zero=False),
    Alloc(v_c, imul(nc_c, v_dim), zero=False),
    Alloc(m_mat, imul(nc_c, chunk_c), zero=False),
    Alloc(y_intra, imul(nc_c, v_dim), zero=False),
    Alloc(b_ch, imul(nc, imul(d_dim, v_dim)), zero=False),
    Alloc(s_in, imul(iadd(nc, ONE), imul(d_dim, v_dim)), zero=False),
    Alloc(y_off, imul(nc_c, v_dim), zero=False),
  ]

  # Step 0a: Parallel chunk decay tables (division-free!)
  c0 = Var("sd0c", I)
  i0 = Var("sd0i", I)
  j0 = Var("sd0j", I)
  p_run = Var("sd0p", elt)
  t_cur = Var("sd0t", I)
  a_val = Var("sd0a", elt)
  cl_a = Cloner({m.t_var.uid: t_cur})
  # Forward cum_a[c, i] = prod_{s=0..i} a[c*C+s]
  fwd_body: list[Stmt] = [
    Assign(t_cur, iadd(imul(c0, chunk_c), i0)),
    If(
      cmp("lt", t_cur, t_dim),
      [Assign(a_val, cl_a.expr(m.a_expr))],
      [Assign(a_val, one_c)],
    ),
    Assign(p_run, ops.prim("mul", (p_run, a_val), elt)),
    Store(cum_a, row_major_index((c0, i0), (nc, chunk_c)), p_run),
  ]
  # Backward tail_a[c, j] = prod_{s=j+1..C-1} a[c*C+s]
  t_bwd = Var("sd0tb", I)
  a_bwd = Var("sd0ab", elt)
  jb = Var("sd0jb", I)
  cl_ab = Cloner({m.t_var.uid: t_bwd})
  bwd_body: list[Stmt] = [
    Assign(jb, isub(Const(SSD_CHUNK - 1, I), i0)),
    Store(tail_a, row_major_index((c0, jb), (nc, chunk_c)), p_run),
    Assign(t_bwd, iadd(imul(c0, chunk_c), jb)),
    If(
      cmp("lt", t_bwd, t_dim),
      [Assign(a_bwd, cl_ab.expr(m.a_expr))],
      [Assign(a_bwd, one_c)],
    ),
    Assign(p_run, ops.prim("mul", (p_run, a_bwd), elt)),
  ]
  # Lower-triangular L[c, i, j] = prod_{s=j+1..i} a[c*C+s] for j <= i else 0
  t_l = Var("sd0tl", I)
  a_l = Var("sd0al", elt)
  jl = Var("sd0jl", I)
  cl_al = Cloner({m.t_var.uid: t_l})
  l_inner: list[Stmt] = [
    Assign(jl, isub(i0, j0)),
    Store(l_mat, row_major_index((c0, i0, jl), (nc, chunk_c, chunk_c)), p_run),
    Assign(t_l, iadd(imul(c0, chunk_c), jl)),
    If(
      cmp("lt", t_l, t_dim),
      [Assign(a_l, cl_al.expr(m.a_expr))],
      [Assign(a_l, one_c)],
    ),
    Assign(p_run, ops.prim("mul", (p_run, a_l), elt)),
  ]
  l_zero_j = Var("sd0jz", I)
  l_outer: list[Stmt] = [
    Assign(p_run, one_c),
    ForRange(j0, ZERO, iadd(i0, ONE), ONE, l_inner, Scope("loop")),
    ForRange(
      l_zero_j,
      iadd(i0, ONE),
      chunk_c,
      ONE,
      [Store(l_mat, row_major_index((c0, i0, l_zero_j), (nc, chunk_c, chunk_c)), zero_c)],
      Scope("loop"),
    ),
  ]
  s0_scope = Scope("loop")
  step0_decay = IndexMap(
    shape=(nc,),
    index=(c0,),
    body=[
      Assign(p_run, one_c),
      ForRange(i0, ZERO, chunk_c, ONE, fwd_body, Scope("loop")),
      Store(ch_dec, c0, p_run),
      Assign(p_run, one_c),
      ForRange(i0, ZERO, chunk_c, ONE, bwd_body, Scope("loop")),
      ForRange(i0, ZERO, chunk_c, ONE, l_outer, Scope("loop")),
      ReleaseScope(s0_scope),
    ],
    scope=s0_scope,
    parallel=True,
    loc=None,
  )

  # Step 0b: Pack Q_c, K_c, K_tail (shape (nc, C, D)) and V_c (shape (nc, C, V))
  pc, pi, pd = Var("pdc", I), Var("pdi", I), Var("pdd", I)
  pt = Var("pdt", I)
  kv_tmp = Var("pkv", elt)
  cl_qk = Cloner({m.t_var.uid: pt, m.d_var.uid: pd})
  pos_d = row_major_index((pc, pi, pd), (nc, chunk_c, d_dim))
  pack_qk_scope = Scope("loop")
  pack_qk = IndexMap(
    shape=(nc, chunk_c, d_dim),
    index=(pc, pi, pd),
    body=[
      Assign(pt, iadd(imul(pc, chunk_c), pi)),
      If(
        cmp("lt", pt, t_dim),
        [
          Store(q_c, pos_d, cl_qk.expr(m.q_expr)),
          Assign(kv_tmp, cl_qk.expr(m.k_expr)),
          Store(k_c, pos_d, kv_tmp),
          Store(k_tail, pos_d, ops.prim("mul", (Load(tail_a, row_major_index((pc, pi), (nc, chunk_c))), kv_tmp), elt)),
        ],
        [
          Store(q_c, pos_d, zero_c),
          Store(k_c, pos_d, zero_c),
          Store(k_tail, pos_d, zero_c),
        ],
      ),
      ReleaseScope(pack_qk_scope),
    ],
    scope=pack_qk_scope,
    parallel=True,
    loc=None,
  )

  vc_c, vc_i, vc_v = Var("pvc", I), Var("pvi", I), Var("pvv", I)
  vt = Var("pvt", I)
  cl_v = Cloner({m.t_var.uid: vt, m.v_var.uid: vc_v})
  pos_v = row_major_index((vc_c, vc_i, vc_v), (nc, chunk_c, v_dim))
  pack_v_scope = Scope("loop")
  pack_v = IndexMap(
    shape=(nc, chunk_c, v_dim),
    index=(vc_c, vc_i, vc_v),
    body=[
      Assign(vt, iadd(imul(vc_c, chunk_c), vc_i)),
      If(
        cmp("lt", vt, t_dim),
        [Store(v_c, pos_v, cl_v.expr(m.v_expr))],
        [Store(v_c, pos_v, zero_c)],
      ),
      ReleaseScope(pack_v_scope),
    ],
    scope=pack_v_scope,
    parallel=True,
    loc=None,
  )

  # Step 1: Intra-chunk quadratic attention-form GEMMs
  # M[c, i, j] = (Q_c[c, i] . K_c[c, j]) * L[c, i, j]
  gemm_m = _batched_gemm_map(m_mat, elt, nc, chunk_c, chunk_c, d_dim, q_c, False, k_c, True, mul_buf=l_mat, loc=None)
  # Y_intra[c, i, v] = M[c, i, :] @ V_c[c, :, v]
  gemm_yi = _batched_gemm_map(y_intra, elt, nc, chunk_c, v_dim, chunk_c, m_mat, False, v_c, False, loc=None)

  # Step 2: Chunk state summaries B_chunk[c, d, v] = K_tail[c, :, d]^T @ V_c[c, :, v]
  gemm_b = _batched_gemm_map(b_ch, elt, nc, d_dim, v_dim, chunk_c, k_tail, True, v_c, False, loc=None)

  # Step 3: Inter-chunk state recurrence S_in[0] = S_0; S_in[c+1] = chunk_decay[c] * S_in[c] + B_chunk[c]
  id0, iv0 = Var("si0d", I), Var("si0v", I)
  s_init_scope = Scope("loop")
  dv_pos0 = row_major_index((id0, iv0), (d_dim, v_dim))
  init_s_in = IndexMap(
    shape=(d_dim, v_dim),
    index=(id0, iv0),
    body=[Store(s_in, dv_pos0, Load(m.state_buf_var, dv_pos0)), ReleaseScope(s_init_scope)],
    scope=s_init_scope,
    parallel=True,
    loc=None,
  )

  sc_c = Var("ssc_c", I)
  sc_d, sc_v = Var("ssc_d", I), Var("ssc_v", I)
  sc_dec = Var("ssc_dec", elt)
  sc_scope = Scope("loop")
  dv_size = imul(d_dim, v_dim)
  cur_off = iadd(imul(sc_c, dv_size), row_major_index((sc_d, sc_v), (d_dim, v_dim)))
  nxt_off = iadd(imul(iadd(sc_c, ONE), dv_size), row_major_index((sc_d, sc_v), (d_dim, v_dim)))
  inter_scan = ForRange(
    sc_c,
    ZERO,
    nc,
    ONE,
    [
      Assign(sc_dec, Load(ch_dec, sc_c)),
      IndexMap(
        shape=(d_dim, v_dim),
        index=(sc_d, sc_v),
        body=[
          Store(
            s_in,
            nxt_off,
            ops.prim("add", (ops.prim("mul", (sc_dec, Load(s_in, cur_off)), elt), Load(b_ch, cur_off)), elt),
          ),
          ReleaseScope(sc_scope),
        ],
        scope=sc_scope,
        parallel=True,
        loc=None,
      ),
    ],
    Scope("loop"),
  )

  # Copy final state S_in[nc] into fresh state buffer so m.state_buf_var stays accurate after loop!
  s_fin_buf = Var("ssd_Sfin", BufType(elt))
  owner.owned.append(s_fin_buf)
  fd, fv = Var("sfd", I), Var("sfv", I)
  s_fin_scope = Scope("loop")
  fin_src = iadd(imul(nc, dv_size), row_major_index((fd, fv), (d_dim, v_dim)))
  fin_dst = row_major_index((fd, fv), (d_dim, v_dim))
  copy_final_s: list[Stmt] = [
    Alloc(s_fin_buf, dv_size, zero=False),
    IndexMap(
      shape=(d_dim, v_dim),
      index=(fd, fv),
      body=[Store(s_fin_buf, fin_dst, Load(s_in, fin_src)), ReleaseScope(s_fin_scope)],
      scope=s_fin_scope,
      parallel=True,
      loc=None,
    ),
    Assign(m.state_buf_var, s_fin_buf),
  ]
  if m.keep_var is not None:
    copy_final_s.extend([Incref(s_fin_buf), Release(m.keep_var), Assign(m.keep_var, s_fin_buf)])

  # Step 4: Off-diagonal state readout Y_off[c, i, v] = Q_c[c, i, :] @ S_in[c, :, v]
  gemm_yo = _batched_gemm_map(y_off, elt, nc, chunk_c, v_dim, d_dim, q_c, False, s_in, False, loc=None)

  # Combine: out[t, v] = Y_intra[c, i, v] + cum_a[c, i] * Y_off[c, i, v]
  ot, ov = Var("sot", I), Var("sov", I)
  oc_v, oi_v = Var("soc", I), Var("soi", I)
  cl_out = Cloner({m.t_var.uid: ot})
  out_pos = iadd(cl_out.expr(m.out_row_offset), ov)
  y_pos = row_major_index((oc_v, oi_v, ov), (nc, chunk_c, v_dim))
  ca_pos = row_major_index((oc_v, oi_v), (nc, chunk_c))
  comb_scope = Scope("loop")
  combine_out = IndexMap(
    shape=(t_dim, v_dim),
    index=(ot, ov),
    body=[
      Assign(oc_v, idiv(ot, chunk_c)),
      Assign(oi_v, imod(ot, chunk_c)),
      Store(
        m.out_buf,
        out_pos,
        ops.prim("add", (Load(y_intra, y_pos), ops.prim("mul", (Load(cum_a, ca_pos), Load(y_off, y_pos)), elt)), elt),
      ),
      ReleaseScope(comb_scope),
    ],
    scope=comb_scope,
    parallel=True,
    loc=None,
  )

  releases = [Release(b) for b in all_tmps]
  ssd_branch: list[Stmt] = [
    *hoisted_checks,
    nc_assign,
    *allocs,
    step0_decay,
    pack_qk,
    pack_v,
    gemm_m,
    gemm_yi,
    gemm_b,
    init_s_in,
    inter_scan,
    gemm_yo,
    combine_out,
    *copy_final_s,
    *releases,
  ]
  return [If(cmp("ge", t_dim, chunk_c), ssd_branch, [loop])]


# ---------------------------------------------------------------------------
# Block driver (also lifts batched IndexMap(*batch): ForRange(t) affine scans!)
# ---------------------------------------------------------------------------


def _try_batched_affine_map(s: IndexMap, owner: Scope) -> list[Stmt] | None:
  """Lift ``IndexMap (*batch): s = init; ForRange(t, 0, T, 1): s = a*s + b; out[...] = s``."""
  if s.ordered or not s.shape or s.scope.owned:
    return None
  clean = [t for t in s.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  if len(clean) != 2 or not isinstance(clean[0], Assign) or not isinstance(clean[1], ForRange):
    return None
  init_st, loop = clean[0], clean[1]
  # T must be independent of batch indices
  b_uids = {v.uid for v in s.index}
  if _expr_uses_uids(loop.stop, b_uids):
    return None
  m = _match_affine_for_range(loop, {init_st.var.uid: init_st.var})
  if m is None or m.state_var.uid != init_st.var.uid:
    return None
  # Emit If(T > AFFINE_CHUNK, batched_tiled_scan, [s])
  tiled = _rewrite_affine_loop(
    loop,
    m,
    owner,
    batch_shape=s.shape,
    batch_index=s.index,
    init_expr=init_st.value,
  )
  return [If(cmp("gt", loop.stop, Const(AFFINE_CHUNK, I)), tiled, [s])]


def _tile_scan_block(
  stmts: list[Stmt],
  facts: IndexFacts,
  outer_vars: dict[int, Var],
  owner: Scope,
) -> list[Stmt]:
  out: list[Stmt] = []
  local_outer = dict(outer_vars)
  for s in stmts:
    facts.record_stmt(s)
    if isinstance(s, Assign):
      local_outer[s.var.uid] = s.var
      out.append(s)
    elif isinstance(s, Alloc):
      local_outer[s.var.uid] = s.var
      out.append(s)
    elif isinstance(s, If):
      out.append(
        If(
          s.cond,
          _tile_scan_block(s.then, facts.fork(), local_outer, owner),
          _tile_scan_block(s.orelse, facts.fork(), local_outer, owner),
        )
      )
    elif isinstance(s, IndexMap):
      lifted = _try_batched_affine_map(s, owner)
      if lifted is not None:
        out.extend(lifted)
      else:
        out.append(s)
    elif isinstance(s, ForRange):
      # 1. Try scalar affine prefix scan (EMA / IIR / Kalman / LRU / GBDT)
      am = _match_affine_for_range(s, local_outer)
      if am is not None:
        out.extend(_rewrite_affine_loop(s, am, owner))
        continue
      # 2. Try bilinear matrix-state recurrence (Mamba-2 SSD / GLA)
      sm = _match_ssd_for_range(s, facts, local_outer)
      if sm is not None:
        out.extend(_rewrite_ssd_loop(s, sm, owner))
        continue
      # Otherwise recurse into loop body (adding loop.var to local_outer!)
      inner_outer = dict(local_outer)
      inner_outer[s.var.uid] = s.var
      out.append(
        ForRange(
          s.var,
          s.start,
          s.stop,
          s.step,
          _tile_scan_block(s.body, facts.fork(), inner_outer, owner),
          s.scope,
          s.simd,
          s.reduction,
        )
      )
    else:
      out.append(s)
  return out


def _param_vars(fn: ir.Function) -> dict[int, Var]:
  out: dict[int, Var] = {}
  for p in fn.params:
    if isinstance(p, Var):
      out[p.uid] = p
    elif isinstance(p, ir.ArrayParam):
      out[p.buf.uid] = p.buf
      if isinstance(p.offset, Var):
        out[p.offset.uid] = p.offset
      for d in p.shape:
        if isinstance(d, Var):
          out[d.uid] = d
      if p.strides is not None:
        for st in p.strides:
          if isinstance(st, Var):
            out[st.uid] = st
  return out


def tile_scans(fn: ir.Function, options) -> None:
  """Law 4 pass: chunkwise parallel affine scan & Mamba-2 SSD state-space duality."""
  if not options.tiling:
    return
  facts = IndexFacts.from_function(fn)
  outer_vars = _param_vars(fn)
  fn.body = _tile_scan_block(fn.body, facts, outer_vars, fn.scope)


__all__ = ["AFFINE_CHUNK", "SSD_CHUNK", "tile_scans"]
