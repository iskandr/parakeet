"""Law 3: Bilinear Contraction Reassociation & MLA Weight Absorption.

Exploits associativity and distributivity of ``(+, *)`` over chained bilinear
tensor contractions::

    (U @ V^T) @ W^T  ==  U @ (V^T @ W^T)

Two general forms are rewritten using an exact FLOP-and-footprint cost model:

1. **General Bilinear Contraction Chains** (LoRA ``(X @ A) @ B`` vs
   ``X @ (A @ B)``, MLP projections, Kronecker chains):
   When producer contraction ``T = contract(U, V)`` feeds a single consumer
   contraction ``Y = contract(T, W)``, compare the cost of the original
   parenthesization against ``M = contract(W, V); Y = contract(M, U)``.
   Statically picks the winner when dimensions are constant, or emits a
   zero-overhead runtime guard ``if reassoc_cost < orig_cost`` so the same
   compiled kernel automatically picks the optimal order for both small-batch
   and large-batch calls.

2. **Latent-Space Attention Absorption (DeepSeek MLA / FlashMLA)**:
   When ``K = C_kv @ W_uk`` and/or ``V = C_kv @ W_uv`` feed an attention block
   ``S = score_mod(Q @ K^T) -> P = softmax(S) -> O = P @ V``, reassociating
   both contractions yields::

       Q_abs = Q @ W_uk^T
       S     = score_mod(Q_abs @ C_kv^T)
       P     = softmax(S)
       O_lat = P @ C_kv
       O     = O_lat @ W_uv

   Because this pass runs immediately before ``monoidal tiling`` (Law 1),
   ``monoidal tiling`` recurses into the absorbed branch and automatically
   fuses ``(Q_abs @ C_kv^T) -> softmax -> (P @ C_kv)`` into FlashAttention /
   FlashDecoding directly over the compressed latent ``C_kv`` — never
   decompressing ``K`` or ``V`` into HBM during autoregressive decode.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ndtypes import BufType, ScalarType
from ..syntax import ir, ops
from ..syntax.clone import Cloner
from ..syntax.ir import (
  Alloc,
  Assign,
  Check,
  Const,
  Expr,
  Fold,
  If,
  IndexMap,
  IndexReduce,
  Load,
  Prim,
  Release,
  ReleaseScope,
  Scope,
  Stmt,
  Store,
  TileStage,
  Var,
)
from ..syntax.ops import I, cast, cmp, iadd, imul, iprod
from .analysis import BufferInfo, IndexFacts, is_pure_expr
from .lowering import row_major_index
from .monoid_tile import (
  _inline_stage,
  _match_ew_row_transform_map,
  _match_row_max_transform_map,
  _match_row_reduce_map,
  _match_row_sum_div_map,
  _match_score_contraction,
  _rebind_stage_batch,
  _resolve_local,
  _ScoreContraction,
)


@dataclass
class _BilinearContraction:
  """A dense row-major contraction ``Y[batch..., r, c] = [scale *] sum_k A(r, k) * B(c, k)``."""

  stmt_idx: int
  stmt: IndexMap
  out_buf: Var
  elt: ScalarType
  batch_shape: tuple[Expr, ...]
  batch_index: tuple[Var, ...]
  dim0: Expr
  dim1: Expr
  k_dim: Expr
  i0: Var
  i1: Var
  a_stage: TileStage  # indexed by (i0, k)
  b_stage: TileStage  # indexed by (i1, k)
  scale: Expr | None
  pre_checks: list[Check]


def _extract_bilinear(s: Stmt, idx: int, facts: IndexFacts) -> _BilinearContraction | None:
  """Extract a pure dot-product contraction (no mask) from ``IndexMap`` ``s``."""
  sc = _match_score_contraction(s, idx, facts)
  if sc is None or sc.score_op != "dot":
    return None
  if sc.mask_cond is not None or sc.mask_buf is not None or sc.mask_fill is not None:
    return None
  return _BilinearContraction(
    stmt_idx=sc.stmt_idx,
    stmt=sc.stmt,
    out_buf=sc.out_buf,
    elt=sc.elt,
    batch_shape=sc.batch_shape,
    batch_index=sc.batch_index,
    dim0=sc.m_dim,
    dim1=sc.n_dim,
    k_dim=sc.k_dim,
    i0=sc.i_var,
    i1=sc.j_var,
    a_stage=sc.q_stage,
    b_stage=sc.k_stage,
    scale=sc.score_scale,
    pre_checks=sc.pre_checks,
  )


def _match_stage_direct_load(
  stage: TileStage,
  buf: Var,
  batch_vars: tuple[Var, ...],
  batch_shape: tuple[Expr, ...],
  dim0: Expr,
  dim1: Expr,
  facts: IndexFacts,
) -> tuple[bool, tuple[Var, ...]] | None:
  """Check if ``stage(p, k)`` is a direct load of dense row-major ``buf``.

  Returns ``(transposed, matched_batch)`` where:
  - ``transposed=False`` means ``buf[batch..., p, k]`` (``p`` is axis 0 of ``buf``, ``k`` is axis 1);
  - ``transposed=True`` means ``buf[batch..., k, p]`` (``k`` is axis 0 of ``buf``, ``p`` is axis 1).
  """
  val_map = {t.var.uid: t.value for t in stage.body if isinstance(t, Assign)}
  ev = _resolve_local(stage.value, val_map)
  if not isinstance(ev, Load) or ev.buf is not buf:
    return None
  p_var, k_var = stage.index
  f = facts.fork()
  for t in stage.body:
    f.record_stmt(t)
  shape_full = (*batch_shape, dim0, dim1)
  if len(batch_vars) == len(batch_shape):
    # Direct order: buf[batch..., p, k]
    idx_direct = row_major_index((*batch_vars, p_var, k_var), shape_full)
    if f.same_expr(ev.index, idx_direct):
      return False, batch_vars
    # Transposed order: buf[batch..., k, p]
    idx_trans = row_major_index((*batch_vars, k_var, p_var), shape_full)
    if f.same_expr(ev.index, idx_trans):
      return True, batch_vars
  # Also allow producer having fewer batch dims (broadcast across leading consumer batch dims)
  if len(batch_shape) == 0:
    idx_direct = row_major_index((p_var, k_var), (dim0, dim1))
    if f.same_expr(ev.index, idx_direct):
      return False, ()
    idx_trans = row_major_index((k_var, p_var), (dim0, dim1))
    if f.same_expr(ev.index, idx_trans):
      return True, ()
  return None


def _stage_uses_buf(stage: TileStage, buf_uid: int) -> bool:
  for s in stage.body:
    if isinstance(s, Assign) and any(isinstance(x, Load) and x.buf.uid == buf_uid for x in ir.walk_expr(s.value)):
      return True
  return any(isinstance(x, Load) and x.buf.uid == buf_uid for x in ir.walk_expr(stage.value))


def _expr_uses_buf(e: Expr | None, buf_uid: int) -> bool:
  if e is None:
    return False
  return any(isinstance(x, Load) and x.buf.uid == buf_uid for x in ir.walk_expr(e))


def _dims_equal(a: Expr, b: Expr, facts: IndexFacts) -> bool:
  return facts.same_expr(a, b)


def _eval_static_cost(expr: Expr) -> int | None:
  if isinstance(expr, Const):
    return int(expr.value)
  if isinstance(expr, Prim) and len(expr.args) == 2:
    a, b = _eval_static_cost(expr.args[0]), _eval_static_cost(expr.args[1])
    if a is None or b is None:
      return None
    if expr.op == "add":
      return a + b
    if expr.op == "mul":
      return a * b
  return None


def _emit_contraction_map(
  out_buf: Var,
  elt: ScalarType,
  batch_shape: tuple[Expr, ...],
  batch_index: tuple[Var, ...],
  dim0: Expr,
  dim1: Expr,
  k_dim: Expr,
  lhs_stage: TileStage,
  rhs_stage: TileStage,
  scale: Expr | None = None,
  mask_cond: Expr | None = None,
  mask_buf: Var | None = None,
  mask_index: Expr | None = None,
  mask_fill: Expr | None = None,
  mask_vars: tuple[Var, Var] | None = None,
  loc=None,
) -> IndexMap:
  """Build a clean ``IndexMap`` computing ``out[batch..., i, j] = [mod(]sum_k lhs(i,k)*rhs(j,k)[)]``."""
  b_vars = tuple(Var("rb", I) for _ in batch_shape)
  i_var = Var("ri", I)
  j_var = Var("rj", I)
  k_var = Var("rk", I)

  lhs_st = _rebind_stage_batch(lhs_stage, batch_index, b_vars)
  rhs_st = _rebind_stage_batch(rhs_stage, batch_index, b_vars)

  r_body: list[Stmt] = []
  lv = _inline_stage(lhs_st, (i_var, k_var), r_body)
  rv = _inline_stage(rhs_st, (j_var, k_var), r_body)
  prod = Var("rx", elt)
  r_body.append(Assign(prod, ops.prim("mul", (lv, rv), elt)))
  a_v, b_v = Var("ra", elt), Var("rb_f", elt)
  res = Var("rred", elt)
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
    index=(k_var,),
    body=r_body,
    folds=[fold],
    scope=Scope("loop"),
    home=scope,
    parallel=False,
    loc=loc,
  )
  val: Expr = res
  if scale is not None:
    val = ops.prim("mul", (val, cast(scale, elt)), elt)
  if mask_fill is not None and (mask_cond is not None or mask_buf is not None):
    subst = {a.uid: b for a, b in zip(batch_index, b_vars)}
    if mask_vars is not None:
      subst[mask_vars[0].uid] = i_var
      subst[mask_vars[1].uid] = j_var
    m_cl = Cloner(subst)
    if mask_buf is not None and mask_index is not None:
      cond_e: Expr = Load(mask_buf, m_cl.expr(mask_index))
    else:
      assert mask_cond is not None
      cond_e = m_cl.expr(mask_cond)
    val = ops.select(cond_e, val, cast(mask_fill, elt))

  shape = (*batch_shape, dim0, dim1)
  idx = (*b_vars, i_var, j_var)
  out_pos = row_major_index(idx, shape)
  m_body: list[Stmt] = [red, Store(out_buf, out_pos, val), ReleaseScope(scope)]
  return IndexMap(shape=shape, index=idx, body=m_body, scope=scope, parallel=True, loc=loc)


def _clone_stmt_list(stmts: list[Stmt]) -> list[Stmt]:
  """Deep-clone statements so each branch of an ``If`` has fresh SSA vars/scopes."""
  cl = Cloner()
  return [cl.stmt(s) for s in stmts]


def _make_buf_stage(
  buf: Var,
  batch_shape: tuple[Expr, ...],
  batch_index: tuple[Var, ...],
  dim0: Expr,
  dim1: Expr,
  transposed: bool = False,
  prefix: str = "rs",
) -> TileStage:
  """Create a ``TileStage`` reading dense row-major ``buf`` at ``(i, k)`` or ``(k, i)``."""
  i_v = Var(f"{prefix}_i", I)
  k_v = Var(f"{prefix}_k", I)
  if transposed:
    idx = row_major_index((*batch_index, k_v, i_v), (*batch_shape, dim1, dim0))
    sh = (dim0, dim1)
  else:
    idx = row_major_index((*batch_index, i_v, k_v), (*batch_shape, dim0, dim1))
    sh = (dim0, dim1)
  val = Load(buf, idx)
  return TileStage(
    buf=Var(f"{prefix}_buf", BufType(buf.type.elt)),
    shape=sh,
    index=(i_v, k_v),
    body=[],
    value=val,
    scope=Scope("stage"),
  )


def _analyze_chain(
  prod: _BilinearContraction,
  cons_batch_shape: tuple[Expr, ...],
  cons_batch_index: tuple[Var, ...],
  cons_dim0: Expr,
  cons_dim1: Expr,
  cons_k_dim: Expr,
  cons_a_stage: TileStage,
  cons_b_stage: TileStage,
  facts: IndexFacts,
):
  """Analyze ``prod -> cons`` where ``cons`` reads ``prod.out_buf`` in one operand.

  Returns a dict describing the three operands ``(U, V, W)`` and exact sizes
  ``(U_dim, K_dim, R_dim, P_dim)`` so::

      Original:     T[u, k] = sum_r U(u, r) * V(k, r)
                    Y[p, u] (or Y[u, p]) = sum_k W(p, k) * T[u, k]
      Reassociated: M[p, r] = sum_k W(p, k) * V(k, r)
                    Y[p, u] (or Y[u, p]) = sum_r M[p, r] * U(u, r)

  Returns ``None`` if ``prod`` and ``cons`` do not form a valid bilinear chain.
  """
  t_buf = prod.out_buf
  if prod.elt is not cons_a_stage.buf.type.elt or prod.scale is not None:
    return None
  # Check which operand of cons reads t_buf (and the other must NOT)
  a_match = _match_stage_direct_load(
    cons_a_stage, t_buf, cons_batch_index, prod.batch_shape, prod.dim0, prod.dim1, facts
  )
  b_match = _match_stage_direct_load(
    cons_b_stage, t_buf, cons_batch_index, prod.batch_shape, prod.dim0, prod.dim1, facts
  )
  if a_match is not None and b_match is None and not _stage_uses_buf(cons_b_stage, t_buf.uid):
    t_is_lhs = True
    t_transposed, prod_batch_in_cons = a_match
    w_stage = cons_b_stage
    p_dim = cons_dim1
    u_dim = cons_dim0
  elif b_match is not None and a_match is None and not _stage_uses_buf(cons_a_stage, t_buf.uid):
    t_is_lhs = False
    t_transposed, prod_batch_in_cons = b_match
    w_stage = cons_a_stage
    p_dim = cons_dim0
    u_dim = cons_dim1
  else:
    return None

  # Rebind prod's stages to cons_batch_index (if prod had batch dims)
  if len(prod.batch_shape) > 0:
    if len(prod.batch_shape) != len(cons_batch_shape):
      return None
    if not all(_dims_equal(a, b, facts) for a, b in zip(prod.batch_shape, cons_batch_shape)):
      return None
    pa = _rebind_stage_batch(prod.a_stage, prod.batch_index, prod_batch_in_cons)
    pb = _rebind_stage_batch(prod.b_stage, prod.batch_index, prod_batch_in_cons)
  else:
    pa = prod.a_stage
    pb = prod.b_stage

  # Match contracted axis k_dim vs free axis u_dim of T
  if not t_transposed:
    # cons accesses T[u, k]: axis 0 is u (dim0), axis 1 is k (dim1)
    if not (_dims_equal(prod.dim0, u_dim, facts) and _dims_equal(prod.dim1, cons_k_dim, facts)):
      return None
    u_stage = pa  # (u, r)
    v_stage = pb  # (k, r)
  else:
    # cons accesses T[k, u]: axis 0 is k (dim0), axis 1 is u (dim1)
    if not (_dims_equal(prod.dim0, cons_k_dim, facts) and _dims_equal(prod.dim1, u_dim, facts)):
      return None
    u_stage = pb  # (u, r)
    v_stage = pa  # (k, r)

  # Verify W does not depend on prod's output buffer
  if _stage_uses_buf(w_stage, t_buf.uid):
    return None

  return {
    "t_is_lhs": t_is_lhs,
    "u_stage": u_stage,  # (u, r)
    "v_stage": v_stage,  # (k, r)
    "w_stage": w_stage,  # (p, k)
    "u_dim": u_dim,
    "k_dim": cons_k_dim,
    "r_dim": prod.k_dim,
    "p_dim": p_dim,
    "batch_shape": cons_batch_shape,
    "batch_index": cons_batch_index,
    "prod_batch_shape": prod.batch_shape,
    "elt": prod.elt,
  }


def _reassoc_cost_exprs(info: dict) -> tuple[Expr, Expr]:
  """Return ``(orig_cost, reassoc_cost)`` MAC counts for a single contraction chain."""
  b_p = iprod(info["prod_batch_shape"]) if info["prod_batch_shape"] else Const(1, I)
  b_c = iprod(info["batch_shape"]) if info["batch_shape"] else Const(1, I)
  # M[batch_c, p, r] = sum_k W(p, k) * V(k, r)
  u, k, r, p = info["u_dim"], info["k_dim"], info["r_dim"], info["p_dim"]
  orig = iadd(imul(imul(b_p, u), imul(k, r)), imul(imul(b_c, p), imul(u, k)))
  reassoc = iadd(imul(imul(b_c, p), imul(k, r)), imul(imul(b_c, p), imul(u, r)))
  return orig, reassoc


def _build_reassoc_pair(
  info: dict,
  y_buf: Var,
  cons_scale: Expr | None,
  cons_mask_cond: Expr | None,
  cons_mask_buf: Var | None,
  cons_mask_index: Expr | None,
  cons_mask_fill: Expr | None,
  cons_mask_vars: tuple[Var, Var] | None,
  owner: Scope | None,
  loc=None,
) -> list[Stmt]:
  """Emit ``M = contract(W, V); Y = contract(M, U); release(M)``."""
  elt: ScalarType = info["elt"]
  b_shape: tuple[Expr, ...] = info["batch_shape"]
  b_idx: tuple[Var, ...] = info["batch_index"]
  p_dim, r_dim, u_dim, k_dim = info["p_dim"], info["r_dim"], info["u_dim"], info["k_dim"]

  m_buf = Var("reassoc_m", BufType(elt))
  if owner is not None:
    owner.owned.append(m_buf)
  m_count = iprod((*b_shape, p_dim, r_dim))
  m_alloc = Alloc(m_buf, m_count, zero=False)

  # Transpose V(k, r) -> V_t(r, k) so _emit_contraction_map contracts over k!
  v_st: TileStage = info["v_stage"]
  v_t = TileStage(
    buf=Var("rvt", BufType(elt)),
    shape=(r_dim, k_dim),
    index=(v_st.index[1], v_st.index[0]),
    body=v_st.body,
    value=v_st.value,
    scope=v_st.scope,
  )
  m_map = _emit_contraction_map(
    out_buf=m_buf,
    elt=elt,
    batch_shape=b_shape,
    batch_index=b_idx,
    dim0=p_dim,
    dim1=r_dim,
    k_dim=k_dim,
    lhs_stage=info["w_stage"],
    rhs_stage=v_t,
    loc=loc,
  )

  m_stage = _make_buf_stage(m_buf, b_shape, b_idx, p_dim, r_dim, transposed=False, prefix="rm")
  u_stage = info["u_stage"]
  if info["t_is_lhs"]:
    # Y[u, p] = sum_r U(u, r) * M(p, r)
    y_map = _emit_contraction_map(
      out_buf=y_buf,
      elt=elt,
      batch_shape=b_shape,
      batch_index=b_idx,
      dim0=u_dim,
      dim1=p_dim,
      k_dim=r_dim,
      lhs_stage=u_stage,
      rhs_stage=m_stage,
      scale=cons_scale,
      mask_cond=cons_mask_cond,
      mask_buf=cons_mask_buf,
      mask_index=cons_mask_index,
      mask_fill=cons_mask_fill,
      mask_vars=cons_mask_vars,
      loc=loc,
    )
  else:
    # Y[p, u] = sum_r M(p, r) * U(u, r)
    y_map = _emit_contraction_map(
      out_buf=y_buf,
      elt=elt,
      batch_shape=b_shape,
      batch_index=b_idx,
      dim0=p_dim,
      dim1=u_dim,
      k_dim=r_dim,
      lhs_stage=m_stage,
      rhs_stage=u_stage,
      scale=cons_scale,
      mask_cond=cons_mask_cond,
      mask_buf=cons_mask_buf,
      mask_index=cons_mask_index,
      mask_fill=cons_mask_fill,
      mask_vars=cons_mask_vars,
      loc=loc,
    )
  return [m_alloc, m_map, y_map, Release(m_buf)]


def _stmt_uses_uid(stmt: Stmt, uid: int) -> bool:
  for s in ir.walk_stmts([stmt]):
    if isinstance(s, Store) and s.buf.uid == uid:
      return True
    if isinstance(s, (ir.Incref, Release)):
      if isinstance(s, Release) and s.var.uid == uid:
        return True
      if isinstance(s, ir.Incref) and any(isinstance(x, Var) and x.uid == uid for x in ir.walk_expr(s.buf)):
        return True
    for e in ir.stmt_exprs(s):
      for x in ir.walk_expr(e):
        if isinstance(x, Var) and x.uid == uid:
          return True
  return False


def _buf_used_outside(stmts: list[Stmt], buf_uid: int, allowed: set[int]) -> bool:
  """Return True if ``buf_uid`` is referenced in any statement index outside ``allowed``."""
  for i, s in enumerate(stmts):
    if i in allowed:
      continue
    if isinstance(s, Alloc) and s.var.uid == buf_uid:
      continue
    if isinstance(s, Release) and s.var.uid == buf_uid:
      continue
    if _stmt_uses_uid(s, buf_uid):
      return True
  return False


def _buf_read_count_in_range(stmts: list[Stmt], buf_uid: int, indices: list[int]) -> bool:
  """Check ``buf_uid`` is not read in any statement outside ``indices`` (ignoring Alloc/Release)."""
  return _buf_used_outside(stmts, buf_uid, set(indices))


def _is_cheap_pure_prelude(s: Stmt) -> bool:
  return isinstance(s, (Assign, Alloc, Check, ir.Comment))


def _scan_next_map(stmts: list[Stmt], pos: int, f: IndexFacts) -> int | None:
  while pos < len(stmts):
    t = stmts[pos]
    if isinstance(t, (ir.Comment, Alloc, Check)):
      f.record_stmt(t)
      pos += 1
      continue
    if isinstance(t, Assign) and not isinstance(t.var.type, BufType) and is_pure_expr(t.value):
      f.record_stmt(t)
      pos += 1
      continue
    if isinstance(t, IndexMap):
      return pos
    return None
  return None


def _match_softmax_chain(
  stmts: list[Stmt], start: int, sc: _ScoreContraction, facts: IndexFacts
) -> tuple[Var, int] | None:
  """Match either fused 2-map or unfused 4-map softmax after score contraction ``sc``.

  Returns ``(p_buf, last_map_idx)`` or ``None``.
  """
  f1 = facts.fork()
  p1 = _scan_next_map(stmts, start, f1)
  if p1 is None:
    return None
  m_buf: Var | None = None
  e_buf: Var | None = None
  e_pos: int | None = None
  f_after_e = f1.fork()
  m_exp = _match_row_max_transform_map(stmts[p1], sc.out_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f1, want_exp=True)
  if m_exp is not None:
    e_buf, m_buf, _ = m_exp
    e_pos = p1
  else:
    m_only = _match_row_reduce_map(stmts[p1], sc.out_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f1, want_op="max")
    if m_only is not None:
      m_buf = m_only
      p1b = _scan_next_map(stmts, p1 + 1, f_after_e)
      if p1b is not None:
        eb = _match_ew_row_transform_map(
          stmts[p1b], sc.out_buf, m_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f_after_e, "exp_sub"
        )
        if eb is not None:
          e_buf = eb
          e_pos = p1b
  if e_buf is None or e_pos is None:
    return None
  f2 = f_after_e.fork()
  p2 = _scan_next_map(stmts, e_pos + 1, f2)
  if p2 is None:
    return None
  m_div = _match_row_sum_div_map(stmts[p2], e_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f2)
  if m_div is not None:
    return m_div[0], p2
  l_only = _match_row_reduce_map(stmts[p2], e_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f2, want_op="add")
  if l_only is not None:
    f3 = f2.fork()
    p2b = _scan_next_map(stmts, p2 + 1, f3)
    if p2b is not None:
      pb = _match_ew_row_transform_map(stmts[p2b], e_buf, l_only, sc.batch_shape, sc.m_dim, sc.n_dim, f3, "div")
      if pb is not None:
        return pb, p2b
  return None


def _try_mla_absorb_at(
  stmts: list[Stmt],
  s_idx: int,
  sc: _ScoreContraction,
  facts: IndexFacts,
  binfo: BufferInfo,
  owner: Scope,
  contractions: dict[int, _BilinearContraction],
) -> tuple[int, int, list[Stmt]] | None:
  """Match MLA attention block ``[K = C@W_uk] -> S = Q@K^T -> softmax -> [V = C@W_uv] -> O = P@V``.

  Reassociates K-projection into Q (``Q_abs = Q @ W_uk^T; S = Q_abs @ C^T``)
  and/or V-projection after attention (``O_lat = P @ C; O = O_lat @ W_uv``)
  guarded by the exact FLOP cost model so Law 1 (``monoidal tiling``) fuses
  the absorbed branch directly into FlashAttention/FlashDecoding over ``C``.
  """
  if sc.score_op != "dot":
    return None
  sm = _match_softmax_chain(stmts, s_idx + 1, sc, facts)
  if sm is None:
    return None
  p_buf, sm_end_idx = sm
  # Scan forward past softmax to find O = P @ V (skipping an optional V = C @ W_uv contraction!)
  o_idx: int | None = None
  v_prod: _BilinearContraction | None = None
  for j in range(sm_end_idx + 1, len(stmts)):
    st = stmts[j]
    if isinstance(st, IndexMap):
      bc = _extract_bilinear(st, j, facts)
      if bc is not None and (_stage_uses_buf(bc.a_stage, p_buf.uid) or _stage_uses_buf(bc.b_stage, p_buf.uid)):
        o_idx = j
        break
      if bc is not None and v_prod is None:
        v_prod = bc
        continue
      if not _expr_uses_buf(None, p_buf.uid):
        # Allow a pure transpose map (like V.T.copy()) before O
        continue
    if not _is_cheap_pure_prelude(st):
      break
  if o_idx is None:
    return None
  o_bc = _extract_bilinear(stmts[o_idx], o_idx, facts)
  if o_bc is None:
    return None

  # Find K producer (before s_idx) if S reads a contraction output buffer
  k_prod: _BilinearContraction | None = None
  k_info: dict | None = None
  for p_uid, cand in contractions.items():
    if cand.stmt_idx >= s_idx:
      continue
    if _stage_uses_buf(sc.q_stage, p_uid) or _stage_uses_buf(sc.k_stage, p_uid):
      info = _analyze_chain(
        cand,
        sc.batch_shape,
        sc.batch_index,
        sc.m_dim,
        sc.n_dim,
        sc.k_dim,
        sc.q_stage,
        sc.k_stage,
        facts,
      )
      if info is not None and not _buf_used_outside(stmts, p_uid, {cand.stmt_idx, s_idx}):
        k_prod = cand
        k_info = info
        break

  # Find V producer (before o_idx, could be before s_idx or between sm_end_idx and o_idx!)
  v_info: dict | None = None
  cand_v_list = list(contractions.values())
  if v_prod is not None:
    cand_v_list.append(v_prod)
  v_prod_matched: _BilinearContraction | None = None
  for cand in cand_v_list:
    if cand.stmt_idx >= o_idx or cand.stmt_idx == s_idx:
      continue
    if k_prod is not None and cand.stmt_idx == k_prod.stmt_idx:
      continue
    p_uid = cand.out_buf.uid
    if _stage_uses_buf(o_bc.a_stage, p_uid) or _stage_uses_buf(o_bc.b_stage, p_uid):
      info = _analyze_chain(
        cand,
        o_bc.batch_shape,
        o_bc.batch_index,
        o_bc.dim0,
        o_bc.dim1,
        o_bc.k_dim,
        o_bc.a_stage,
        o_bc.b_stage,
        facts,
      )
      if info is not None and not _buf_used_outside(stmts, p_uid, {cand.stmt_idx, o_idx}):
        v_prod_matched = cand
        v_info = info
        break

  if k_info is None and v_info is None:
    return None

  # Ensure S and P buffers are not escaped outside [start .. o_idx]
  start_idx = s_idx
  if k_prod is not None and k_prod.stmt_idx < start_idx:
    start_idx = k_prod.stmt_idx
  if v_prod_matched is not None and v_prod_matched.stmt_idx < start_idx:
    start_idx = v_prod_matched.stmt_idx
  span_set = set(range(start_idx, o_idx + 1))

  # Collect any allocs/assigns of intermediate buffers inside [start_idx .. o_idx]
  # Keep o_bc.out_buf's Alloc OUTSIDE the If (so both branches write into o_bc.out_buf)!
  o_buf_uid = o_bc.out_buf.uid
  k_uid = k_prod.out_buf.uid if k_prod is not None else -1
  v_uid = v_prod_matched.out_buf.uid if v_prod_matched is not None else -1
  skip_in_abs = {s_idx, o_idx}
  if k_prod is not None:
    skip_in_abs.add(k_prod.stmt_idx)
  if v_prod_matched is not None:
    skip_in_abs.add(v_prod_matched.stmt_idx)

  # Make sure no other statement in [start_idx .. o_idx] writes to external live buffers
  # (only K, V, S, softmax temps, O, or hoisted transpose helpers).
  hoisted_before: list[Stmt] = []
  inner_orig: list[Stmt] = []
  inner_abs: list[Stmt] = []

  # Any statement in [start_idx .. o_idx] that defines a scalar/buffer used OUTSIDE [start_idx .. o_idx]
  # (such as o_bc.out_buf's Alloc or transpose maps like bt_21/bt_39) must be hoisted before the If!
  def _hoist_needed(idx_s: int, st: Stmt) -> bool:
    if isinstance(st, Alloc) and st.var.uid == o_buf_uid:
      return True
    if idx_s in skip_in_abs:
      return False
    # Check if st defines a var used outside span_set, or is a pure transpose/setup used by k_info/v_info
    if isinstance(st, (Assign, Check)):
      return True
    if isinstance(st, Alloc) and st.var.uid not in (k_uid, v_uid, sc.out_buf.uid, p_buf.uid):
      # Intermediate softmax buffer (like arr_113) vs independent buffer (like bt_39)
      if _buf_used_outside(stmts, st.var.uid, span_set):
        return True
      # If it's used by k_info or v_info stages, hoist it before the If!
      for inf in (k_info, v_info):
        if inf is not None:
          if any(_stage_uses_buf(inf[key], st.var.uid) for key in ("u_stage", "v_stage", "w_stage")):
            return True
    if isinstance(st, IndexMap):
      # If this map computes a buffer used by k_info/v_info (e.g. bt_39 = w_uv.T), hoist it!
      written = {t.buf.uid for t in st.body if isinstance(t, Store)}
      if written & {k_uid, v_uid, sc.out_buf.uid, p_buf.uid}:
        return False
      for w_uid in written:
        if _buf_used_outside(stmts, w_uid, span_set):
          return True
        for inf in (k_info, v_info):
          if inf is not None and any(_stage_uses_buf(inf[key], w_uid) for key in ("u_stage", "v_stage", "w_stage")):
            return True
    return False

  # Build hoisted_before, inner_orig, inner_abs
  hoisted_indices: set[int] = set()
  for idx_s in range(start_idx, o_idx + 1):
    if _hoist_needed(idx_s, stmts[idx_s]):
      hoisted_indices.add(idx_s)
      hoisted_before.append(stmts[idx_s])

  # Ensure any Alloc for a hoisted buffer is also hoisted
  hoisted_bufs: set[int] = set()
  for idx_s in hoisted_indices:
    st = stmts[idx_s]
    if isinstance(st, IndexMap):
      for t in st.body:
        if isinstance(t, Store):
          hoisted_bufs.add(t.buf.uid)
  for idx_s in range(start_idx, o_idx + 1):
    if idx_s not in hoisted_indices:
      st = stmts[idx_s]
      if isinstance(st, Alloc) and st.var.uid in hoisted_bufs:
        hoisted_indices.add(idx_s)
  # Rebuild hoisted_before in original order
  hoisted_before = [stmts[idx_s] for idx_s in range(start_idx, o_idx + 1) if idx_s in hoisted_indices]

  # Verify no statement in [start_idx..o_idx] outside hoisted_indices defines a buffer used after o_idx
  # (other than o_buf_uid, which was hoisted)
  for idx_s in range(start_idx, o_idx + 1):
    if idx_s in hoisted_indices:
      continue
    st = stmts[idx_s]
    if isinstance(st, Alloc) and st.var.uid != o_buf_uid:
      if _buf_used_outside(stmts, st.var.uid, span_set):
        return None

  # Build original branch
  for idx_s in range(start_idx, o_idx + 1):
    if idx_s not in hoisted_indices:
      inner_orig.append(stmts[idx_s])

  # Build absorbed branch (defer owner.owned registration until after optional branch clone;
  # defer Release(m_buf) to the end of the branch so S -> softmax -> O_lat stay adjacent for Law 1!)
  deferred_releases: list[Stmt] = []
  for idx_s in range(start_idx, o_idx + 1):
    if idx_s in hoisted_indices:
      continue
    st = stmts[idx_s]
    if k_prod is not None and (idx_s == k_prod.stmt_idx or (isinstance(st, Alloc) and st.var.uid == k_uid)):
      continue
    if v_prod_matched is not None and (
      idx_s == v_prod_matched.stmt_idx or (isinstance(st, Alloc) and st.var.uid == v_uid)
    ):
      continue
    if idx_s == s_idx:
      if k_info is not None:
        pair_k = _build_reassoc_pair(
          k_info,
          sc.out_buf,
          sc.score_scale,
          sc.mask_cond,
          sc.mask_buf,
          sc.mask_index,
          sc.mask_fill,
          (sc.i_var, sc.j_var),
          None,
          loc=sc.stmt.loc,
        )
        for ps in pair_k:
          if isinstance(ps, Release):
            deferred_releases.append(ps)
          else:
            inner_abs.append(ps)
      else:
        inner_abs.append(st)
      continue
    if idx_s == o_idx:
      if v_info is not None:
        pair_v = _build_reassoc_pair(
          v_info,
          o_bc.out_buf,
          o_bc.scale,
          None,
          None,
          None,
          None,
          None,
          None,
          loc=o_bc.stmt.loc,
        )
        for ps in pair_v:
          if isinstance(ps, Release):
            deferred_releases.append(ps)
          else:
            inner_abs.append(ps)
      else:
        inner_abs.append(st)
      continue
    inner_abs.append(st)
  inner_abs.extend(deferred_releases)

  # Total cost comparison across absorbed K and/or V contractions
  orig_total: Expr | None = None
  abs_total: Expr | None = None
  for inf in (k_info, v_info):
    if inf is not None:
      oc, rc = _reassoc_cost_exprs(inf)
      orig_total = oc if orig_total is None else iadd(orig_total, oc)
      abs_total = rc if abs_total is None else iadd(abs_total, rc)
  assert orig_total is not None and abs_total is not None

  s_orig = _eval_static_cost(orig_total)
  s_abs = _eval_static_cost(abs_total)
  if s_orig is not None and s_abs is not None:
    if s_abs < s_orig:
      for s_c in inner_abs:
        if isinstance(s_c, Alloc) and s_c.var not in owner.owned:
          owner.owned.append(s_c.var)
      return start_idx, o_idx + 1, [*hoisted_before, *inner_abs]
    return None

  # Dynamic runtime cost selection: If(abs_cost < orig_cost, absorbed, original)
  # Clone inner_abs so SSA vars/scopes are completely disjoint from inner_orig!
  # Note: keep shared buffer vars (like sc.out_buf, p_buf, o_bc.out_buf, hoisted bufs) un-cloned!
  # Easiest: allocate fresh intermediate buffers inside inner_abs via Cloner on inner_abs!
  inner_abs_cloned = _clone_stmt_list(inner_abs)
  # Ensure any newly cloned Alloc vars are owned by owner
  for s_c in inner_abs_cloned:
    if isinstance(s_c, Alloc) and s_c.var not in owner.owned:
      owner.owned.append(s_c.var)
  cond = cmp("lt", abs_total, orig_total)
  return start_idx, o_idx + 1, [*hoisted_before, If(cond, inner_abs_cloned, inner_orig)]


def _try_general_chain_at(
  stmts: list[Stmt],
  c_idx: int,
  cons: _BilinearContraction,
  facts: IndexFacts,
  owner: Scope,
  contractions: dict[int, _BilinearContraction],
) -> tuple[int, int, list[Stmt]] | None:
  """Rewrite a general 2-contraction chain ``T = U @ V^T; Y = W @ T^T`` when cheaper."""
  for p_uid, prod in contractions.items():
    if prod.stmt_idx >= c_idx:
      continue
    if not (_stage_uses_buf(cons.a_stage, p_uid) or _stage_uses_buf(cons.b_stage, p_uid)):
      continue
    info = _analyze_chain(
      prod,
      cons.batch_shape,
      cons.batch_index,
      cons.dim0,
      cons.dim1,
      cons.k_dim,
      cons.a_stage,
      cons.b_stage,
      facts,
    )
    if info is None:
      continue
    if _buf_used_outside(stmts, p_uid, {prod.stmt_idx, c_idx}):
      continue

    orig_c, reassoc_c = _reassoc_cost_exprs(info)
    s_orig = _eval_static_cost(orig_c)
    s_reassoc = _eval_static_cost(reassoc_c)
    if s_orig is not None and s_reassoc is not None and s_reassoc >= s_orig:
      continue

    start_idx = prod.stmt_idx
    end_idx = c_idx + 1
    hoisted: list[Stmt] = []
    orig_body: list[Stmt] = []
    for k in range(start_idx, end_idx):
      st = stmts[k]
      if k == prod.stmt_idx or k == c_idx:
        orig_body.append(st)
      elif isinstance(st, Alloc) and st.var.uid == p_uid:
        orig_body.append(st)
      else:
        hoisted.append(st)

    reassoc_body = _build_reassoc_pair(
      info,
      cons.out_buf,
      cons.scale,
      None,
      None,
      None,
      None,
      None,
      owner,
      loc=cons.stmt.loc,
    )
    if s_orig is not None and s_reassoc is not None:
      return start_idx, end_idx, [*hoisted, *reassoc_body]
    cond = cmp("lt", reassoc_c, orig_c)
    return start_idx, end_idx, [*hoisted, If(cond, reassoc_body, orig_body)]
  return None


def _reassociate_block(
  stmts: list[Stmt],
  facts: IndexFacts,
  binfo: BufferInfo,
  owner: Scope,
) -> list[Stmt]:
  # First recurse into nested control flow
  out_stmts: list[Stmt] = []
  for s in stmts:
    if isinstance(s, If):
      out_stmts.append(
        If(
          s.cond,
          _reassociate_block(s.then, facts.fork(), binfo, owner),
          _reassociate_block(s.orelse, facts.fork(), binfo, owner),
        )
      )
    elif isinstance(s, ir.ForRange):
      out_stmts.append(
        ir.ForRange(
          s.var,
          s.start,
          s.stop,
          s.step,
          _reassociate_block(s.body, facts.fork(), binfo, owner),
          s.scope,
          s.simd,
          s.reduction,
        )
      )
    else:
      out_stmts.append(s)

  # Scan block for MLA attention absorption first (so K+V absorb together), then general chains
  work = out_stmts
  changed = True
  while changed:
    changed = False
    f = facts.fork()
    contractions: dict[int, _BilinearContraction] = {}
    for idx, s in enumerate(work):
      f.record_stmt(s)
      if not isinstance(s, IndexMap):
        continue
      # 1. Try MLA attention absorption anchored at score contraction S
      sc = _match_score_contraction(s, idx, f)
      if sc is not None:
        mla = _try_mla_absorb_at(work, idx, sc, f, binfo, owner, contractions)
        if mla is not None:
          lo, hi, repl = mla
          work = work[:lo] + repl + work[hi:]
          changed = True
          break
      # 2. Try general 2-contraction chain (LoRA, MLP, Kronecker)
      bc = _extract_bilinear(s, idx, f)
      if bc is not None:
        feeds_softmax = sc is not None and _match_softmax_chain(work, idx + 1, sc, f) is not None
        if not feeds_softmax:
          gen = _try_general_chain_at(work, idx, bc, f, owner, contractions)
          if gen is not None:
            lo, hi, repl = gen
            work = work[:lo] + repl + work[hi:]
            changed = True
            break
        contractions[bc.out_buf.uid] = bc
  return work


def reassociate_contractions(fn: ir.Function, options) -> None:
  """Law 3 pass: cost-model-driven contraction reassociation & MLA weight absorption."""
  if not options.fusion:
    return
  facts = IndexFacts.from_function(fn)
  binfo = BufferInfo.analyze(fn)
  fn.body = _reassociate_block(fn.body, facts, binfo, fn.scope)


__all__ = ["reassociate_contractions"]
