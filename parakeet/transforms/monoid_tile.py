"""Law 1 + Law 5: 2-Level Rescaled Semiring Monoid Tile Fusion.

Eliminates ``O(N^2)`` and ``O(N * V)`` HBM materialization by fusing tiled
contractions (dot products or pairwise distances) directly into associative
rescaled-monoid reductions over tiles:

* **FlashAttention & FlashDecoding & Pairwise Kernel Smoothing**
  (``kind="flash_monoid"``)::

      S = score_mod(contract_k(Q, K), mask)      # O(B * H * M * N)  -- eliminated!
      P = softmax(S, axis=-1)                    # O(B * H * M * N)  -- eliminated!
      O = contract_j(P, V)                       # O(B * H * M * D)

  Fused into a single ``IndexTile`` carrying the online rescaled monoid
  ``(m, l, O)`` with Law 5 affine tile-interval mask pruning (causal, sliding
  window, spatial cutoff, or boolean-mask ballot skip) and automatic Split-K
  FlashDecoding when the query grid is small relative to device SMs.

* **Cut Cross-Entropy & Pairwise LogSumExp** (``kind="monoid_lse"``)::

      S = contract_k(H, W)                       # O(M * V)  -- eliminated!
      lse = logsumexp(S, axis=-1)                # O(M)
      picked = S[arange(M), targets]             # optional O(M) target gather

  Fused into a single ``IndexTile`` carrying the online ``(m, l)`` monoid and
  target dot product without materializing the ``M x V`` score matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ndtypes import Bool, BufType, ScalarType
from ..syntax import ir, ops
from ..syntax.clone import Cloner, defined
from ..syntax.ir import (
  Assign,
  Check,
  Const,
  Expr,
  ForRange,
  If,
  IndexMap,
  IndexReduce,
  IndexTile,
  Load,
  MonoidalTilePlan,
  Prim,
  ReleaseScope,
  Scope,
  Stmt,
  Store,
  TileStage,
  Var,
)
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imod, imul, iprod
from .analysis import BufferInfo, IndexFacts, TileInterval, is_pure_expr, tile_predicate_guards
from .lowering import row_major_index
from .simplify import drop_buffers


@dataclass
class _ScoreContraction:
  """Extracted contraction ``S[batch..., i, j] = score_mod(reduce_k f(Q[i,k], K[j,k]))``."""

  stmt_idx: int
  stmt: IndexMap
  out_buf: Var
  elt: ScalarType
  batch_shape: tuple[Expr, ...]
  batch_index: tuple[Var, ...]
  m_dim: Expr
  n_dim: Expr
  k_dim: Expr
  i_var: Var
  j_var: Var
  q_stage: TileStage
  k_stage: TileStage
  score_op: str  # "dot" | "sqdist"
  score_scale: Expr | None
  mask_cond: Expr | None
  mask_buf: Var | None
  mask_index: Expr | None
  mask_fill: Expr | None
  pre_checks: list[Check]
  bias_stage: TileStage | None = None
  ipa_xq_stage: TileStage | None = None
  ipa_xk_stage: TileStage | None = None
  ipa_c_dim: Expr | None = None
  ipa_gamma: Expr | None = None


def _expr_uses_uids(e: Expr, uids: set[int]) -> bool:
  return any(isinstance(x, Var) and x.uid in uids for x in ir.walk_expr(e))


def _resolve_local(e: Expr, locals_map: dict[int, Expr], budget: int = 16) -> Expr:
  if budget <= 0:
    return e
  if isinstance(e, Var) and e.uid in locals_map:
    return _resolve_local(locals_map[e.uid], locals_map, budget - 1)
  if isinstance(e, Prim):
    return Prim(e.op, tuple(_resolve_local(a, locals_map, budget - 1) for a in e.args), e.type)
  if isinstance(e, ir.Cast):
    return ir.Cast(_resolve_local(e.arg, locals_map, budget - 1), e.type)
  if isinstance(e, Load):
    return Load(e.buf, _resolve_local(e.index, locals_map, budget - 1))
  if isinstance(e, ir.RtCall):
    return ir.RtCall(e.name, tuple(_resolve_local(a, locals_map, budget - 1) for a in e.args), e.type, e.pure)
  return e


def _build_stage(
  e: Expr,
  idx_vars: tuple[Var, ...],
  shape: tuple[Expr, ...],
  local_assigns: dict[int, Assign],
  order_map: dict[int, int],
  name_prefix: str,
) -> TileStage:
  """Package pure expression ``e`` (with any needed local assigns) into a :class:`TileStage`."""
  dep_uids: set[int] = set()

  def collect(ex: Expr) -> None:
    for x in ir.walk_expr(ex):
      if isinstance(x, Var) and x.uid in local_assigns and x.uid not in dep_uids:
        dep_uids.add(x.uid)
        collect(local_assigns[x.uid].value)

  collect(e)
  dep_stmts = [local_assigns[u] for u in sorted(dep_uids, key=lambda u: order_map[u])]
  new_idx = tuple(Var(f"{name_prefix}_{k}", I) for k in range(len(idx_vars)))
  cl = Cloner({v.uid: nv for v, nv in zip(idx_vars, new_idx)})
  for st in dep_stmts:
    cl.var(st.var)
  body = [cl.stmt(st) for st in dep_stmts]
  val = cl.expr(e)
  sbuf = Var(f"st_{name_prefix}", BufType(e.type.strong))
  return TileStage(sbuf, shape, new_idx, body, val, Scope("stage"))


def _parse_scaled_var(e: Expr, target: Var, inner_uids: set[int], elt: ScalarType) -> tuple[bool, Expr | None]:
  """Match ``target``, ``target * s``, ``s * target``, ``target / s``, or ``-target``."""
  if e is target:
    return True, None
  if isinstance(e, Prim) and e.op == "mul" and len(e.args) == 2:
    if e.args[0] is target and not _expr_uses_uids(e.args[1], inner_uids):
      return True, e.args[1]
    if e.args[1] is target and not _expr_uses_uids(e.args[0], inner_uids):
      return True, e.args[0]
  if isinstance(e, Prim) and e.op == "div" and len(e.args) == 2:
    if e.args[0] is target and not _expr_uses_uids(e.args[1], inner_uids):
      return True, ops.prim("div", (Const(1.0, elt), e.args[1]), elt)
  if isinstance(e, Prim) and e.op == "neg" and len(e.args) == 1 and e.args[0] is target:
    return True, Const(-1.0, elt)
  return False, None


def _analyze_red_contraction(
  red: IndexReduce,
  i_var: Var,
  j_var: Var,
  s_index_uids: set[int],
  base_assigns: list[Assign],
) -> tuple[str, Expr, TileStage, TileStage, Var, dict[int, Assign], dict[int, int]] | None:
  """Extract ``(score_op, k_dim, q_stage, k_stage, result_var, assign_map, order_map)``
  from a 1D float add-reduction computing a dot product or squared distance."""
  if len(red.shape) != 1 or len(red.folds) != 1 or red.where is not None or red.scope.owned:
    return None
  fold = red.folds[0]
  if fold.op != "add" or fold.combine or fold.carries or fold.a.type.kind != "f":
    return None
  if not (isinstance(fold.init, Const) and float(fold.init.value) == 0.0):
    return None
  if any(not isinstance(t, (Assign, ir.Comment)) for t in red.body):
    return None
  k_var = red.index[0]
  k_dim = red.shape[0]
  if _expr_uses_uids(k_dim, s_index_uids):
    return None

  all_assigns = [*base_assigns, *(t for t in red.body if isinstance(t, Assign))]
  assign_map = {t.var.uid: t for t in all_assigns}
  order_map = {t.var.uid: k for k, t in enumerate(all_assigns)}
  val_map = {t.var.uid: t.value for t in all_assigns}

  fv = _resolve_local(fold.value, val_map)
  score_op = ""
  lhs_e: Expr | None = None
  rhs_e: Expr | None = None
  if isinstance(fv, Prim) and fv.op == "mul" and len(fv.args) == 2:
    a, b = fv.args
    if (
      isinstance(a, Prim)
      and a.op == "sub"
      and len(a.args) == 2
      and isinstance(b, Prim)
      and b.op == "sub"
      and len(b.args) == 2
      and ir.same(a.args[0], b.args[0])
      and ir.same(a.args[1], b.args[1])
    ):
      score_op = "sqdist"
      lhs_e, rhs_e = a.args
    else:
      score_op = "dot"
      lhs_e, rhs_e = a, b
  elif (
    isinstance(fv, Prim)
    and fv.op == "pow"
    and len(fv.args) == 2
    and isinstance(fv.args[1], Const)
    and float(fv.args[1].value) == 2.0
    and isinstance(fv.args[0], Prim)
    and fv.args[0].op == "sub"
    and len(fv.args[0].args) == 2
  ):
    score_op = "sqdist"
    lhs_e, rhs_e = fv.args[0].args
  if not score_op or lhs_e is None or rhs_e is None:
    return None

  def classify(e: Expr) -> str | None:
    if not is_pure_expr(e) or not any(isinstance(x, Load) for x in ir.walk_expr(e)):
      return None
    ui = _expr_uses_uids(e, {i_var.uid})
    uj = _expr_uses_uids(e, {j_var.uid})
    uk = _expr_uses_uids(e, {k_var.uid})
    if not uk:
      return None
    if ui and not uj:
      return "row"
    if uj and not ui:
      return "col"
    return None

  c1, c2 = classify(lhs_e), classify(rhs_e)
  if (c1, c2) == ("row", "col"):
    q_expr, k_expr = lhs_e, rhs_e
  elif (c1, c2) == ("col", "row"):
    q_expr, k_expr = rhs_e, lhs_e
  else:
    return None

  q_stage = _build_stage(q_expr, (i_var, k_var), (Const(0, I), k_dim), assign_map, order_map, "q")
  k_stage = _build_stage(k_expr, (j_var, k_var), (Const(0, I), k_dim), assign_map, order_map, "k")
  return score_op, k_dim, q_stage, k_stage, fold.result, assign_map, order_map


def _match_score_contraction(s: Stmt, idx: int, facts: IndexFacts) -> _ScoreContraction | None:
  """Recognize an ``IndexMap`` over ``(*batch, M, N)`` computing a 1D contraction
  (dot product or squared distance), optional 3D IPA squared-distance bias,
  optional additive pair bias ``B(i, j)``, optional scale, and optional mask."""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) < 2 or s.scope.owned:
    return None
  body = [
    t for t in s.body if not (isinstance(t, ReleaseScope) and t.scope is s.scope) and not isinstance(t, ir.Comment)
  ]
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  if len(reds) not in (1, 2):
    return None

  first_r = reds[0]
  last_r = reds[-1]
  before = body[:first_r]
  between = body[first_r + 1 : last_r] if len(reds) == 2 else []
  after = body[last_r + 1 :]
  if any(not isinstance(t, (Assign, Check, ir.Comment)) for t in before):
    return None
  if any(not isinstance(t, (Assign, Check, ir.Comment)) for t in between):
    return None
  stores = [t for t in after if isinstance(t, Store)]
  if len(stores) != 1 or any(not isinstance(t, (Assign, Store, ir.Comment)) for t in after):
    return None
  store = stores[0]
  if store.atomic is not None:
    return None

  expected_idx = row_major_index(s.index, s.shape)
  local_facts = facts.fork()
  for t in before + between + after:
    local_facts.record_stmt(t)
  if not local_facts.same_expr(store.index, expected_idx):
    return None

  batch_shape = s.shape[:-2]
  batch_index = s.index[:-2]
  m_dim, n_dim = s.shape[-2], s.shape[-1]
  i_var, j_var = s.index[-2], s.index[-1]
  s_index_uids = {v.uid for v in s.index}

  base_assigns_1 = [t for t in before if isinstance(t, Assign)]
  red1 = body[reds[0]]
  assert isinstance(red1, IndexReduce)
  c1 = _analyze_red_contraction(red1, i_var, j_var, s_index_uids, base_assigns_1)
  if c1 is None:
    return None
  elt = red1.folds[0].a.type
  if store.buf.type.elt is not elt:
    return None

  ipa_xq_stage: TileStage | None = None
  ipa_xk_stage: TileStage | None = None
  ipa_c_dim: Expr | None = None
  ipa_gamma: Expr | None = None
  sq_result: Var | None = None

  if len(reds) == 1:
    score_op, k_dim, q_stage, k_stage, dot_result, assign_map, order_map = c1
    k_uids = {red1.index[0].uid}
  else:
    base_assigns_2 = [t for t in before + between if isinstance(t, Assign)]
    red2 = body[reds[1]]
    assert isinstance(red2, IndexReduce)
    c2 = _analyze_red_contraction(red2, i_var, j_var, s_index_uids, base_assigns_2)
    if c2 is None or red2.folds[0].a.type is not elt:
      return None
    if c1[0] == "dot" and c2[0] == "sqdist":
      score_op, k_dim, q_stage, k_stage, dot_result, assign_map, order_map = c1
      _, ipa_c_dim, ipa_xq_stage, ipa_xk_stage, sq_result, _, _ = c2
    elif c2[0] == "dot" and c1[0] == "sqdist":
      score_op, k_dim, q_stage, k_stage, dot_result, assign_map, order_map = c2
      _, ipa_c_dim, ipa_xq_stage, ipa_xk_stage, sq_result, _, _ = c1
    else:
      return None
    k_uids = {red1.index[0].uid, red2.index[0].uid}
    ipa_xq_stage = TileStage(
      Var("st_xq", ipa_xq_stage.buf.type),
      (m_dim, ipa_c_dim),
      ipa_xq_stage.index,
      ipa_xq_stage.body,
      ipa_xq_stage.value,
      Scope("stage"),
    )
    ipa_xk_stage = TileStage(
      Var("st_xk", ipa_xk_stage.buf.type),
      (n_dim, ipa_c_dim),
      ipa_xk_stage.index,
      ipa_xk_stage.body,
      ipa_xk_stage.value,
      Scope("stage"),
    )

  # Fix stage shapes with actual m_dim / n_dim
  q_stage = TileStage(q_stage.buf, (m_dim, k_dim), q_stage.index, q_stage.body, q_stage.value, Scope("stage"))
  k_stage = TileStage(k_stage.buf, (n_dim, k_dim), k_stage.index, k_stage.body, k_stage.value, Scope("stage"))

  # Parse epilogue: store.value == [select(mask, core, fill)]
  outer_assigns = [t for t in before + between + after if isinstance(t, Assign)]
  outer_assign_map = {t.var.uid: t for t in outer_assigns}
  outer_order_map = {t.var.uid: idx_o for idx_o, t in enumerate(outer_assigns)}
  after_vals = {t.var.uid: t.value for t in outer_assigns}
  sv = _resolve_local(store.value, after_vals)
  mask_cond: Expr | None = None
  mask_buf: Var | None = None
  mask_index: Expr | None = None
  mask_fill: Expr | None = None
  core = sv
  if isinstance(sv, Prim) and sv.op == "select" and len(sv.args) == 3:
    cond_e, then_e, else_e = sv.args
    if _expr_uses_uids(then_e, {dot_result.uid}) and not _expr_uses_uids(else_e, {dot_result.uid}):
      core = then_e
      mask_fill = else_e
      if isinstance(cond_e, Load) and cond_e.buf.type.elt is Bool:
        mask_buf = cond_e.buf
        mask_index = cond_e.index
      elif is_pure_expr(cond_e) and not any(isinstance(x, Load) for x in ir.walk_expr(cond_e)):
        mask_cond = cond_e
      else:
        return None
    else:
      return None

  inner_uids = {i_var.uid, j_var.uid, *k_uids, *(v.uid for v in batch_index)} | set(assign_map)
  bias_expr: Expr | None = None

  # Peel optional additive/subtractive bias and/or IPA sqdist term around dot_result
  if sq_result is not None:
    # Expect core == dot_part - gamma * sq_part  (or dot_part + (-gamma) * sq_part), plus optional bias_expr
    terms: list[tuple[int, Expr]] = []

    def _collect_add_terms(ex: Expr, sign: int) -> None:
      if isinstance(ex, Prim) and ex.op == "add" and len(ex.args) == 2:
        _collect_add_terms(ex.args[0], sign)
        _collect_add_terms(ex.args[1], sign)
      elif isinstance(ex, Prim) and ex.op == "sub" and len(ex.args) == 2:
        _collect_add_terms(ex.args[0], sign)
        _collect_add_terms(ex.args[1], -sign)
      else:
        terms.append((sign, ex))

    _collect_add_terms(core, 1)
    dot_term: tuple[int, Expr] | None = None
    sq_term: tuple[int, Expr] | None = None
    extra_bias: list[tuple[int, Expr]] = []
    for sgn, tm in terms:
      u_dot = _expr_uses_uids(tm, {dot_result.uid})
      u_sq = _expr_uses_uids(tm, {sq_result.uid})
      if u_dot and not u_sq and dot_term is None:
        dot_term = (sgn, tm)
      elif u_sq and not u_dot and sq_term is None:
        sq_term = (sgn, tm)
      elif not u_dot and not u_sq and is_pure_expr(tm) and not _expr_uses_uids(tm, k_uids):
        extra_bias.append((sgn, tm))
      else:
        return None
    if dot_term is None or sq_term is None:
      return None
    ok_d, sc_d = _parse_scaled_var(dot_term[1], dot_result, inner_uids, elt)
    ok_s, sc_s = _parse_scaled_var(sq_term[1], sq_result, inner_uids, elt)
    if not ok_d or not ok_s:
      return None
    if dot_term[0] < 0:
      sc_d = Const(-1.0, elt) if sc_d is None else ops.prim("neg", (sc_d,), elt)
    score_scale = sc_d
    # Score formula uses: dot * score_scale - ipa_gamma * sqdist
    raw_sq = Const(1.0, elt) if sc_s is None else sc_s
    ipa_gamma = raw_sq if sq_term[0] < 0 else ops.prim("neg", (raw_sq,), elt)
    for sgn, tm in extra_bias:
      piece = tm if sgn > 0 else ops.prim("neg", (tm,), elt)
      bias_expr = piece if bias_expr is None else ops.prim("add", (bias_expr, piece), elt)
  else:
    # Single contraction: peel optional + bias / - bias
    if isinstance(core, Prim) and core.op in ("add", "sub") and len(core.args) == 2:
      op_ab = core.op
      a, b = core.args
      ua = _expr_uses_uids(a, {dot_result.uid})
      ub = _expr_uses_uids(b, {dot_result.uid})
      if ua and not ub and is_pure_expr(b) and not _expr_uses_uids(b, k_uids):
        core = a
        bias_expr = b if b.type is elt else cast(b, elt)
        if op_ab == "sub":
          bias_expr = ops.prim("neg", (bias_expr,), elt)
      elif op_ab == "add" and ub and not ua and is_pure_expr(a) and not _expr_uses_uids(a, k_uids):
        core = b
        bias_expr = a if a.type is elt else cast(a, elt)
    ok_d, score_scale = _parse_scaled_var(core, dot_result, inner_uids, elt)
    if not ok_d:
      return None

  if mask_fill is not None and _expr_uses_uids(mask_fill, inner_uids):
    return None

  pre_checks = [t for t in before + between if isinstance(t, Check)]
  for chk in pre_checks:
    if any(_expr_uses_uids(e, inner_uids) for e in (chk.cond, *chk.args)):
      return None

  bias_stage: TileStage | None = None
  if bias_expr is not None:
    if _expr_uses_uids(bias_expr, {dot_result.uid} | k_uids | ({sq_result.uid} if sq_result is not None else set())):
      return None
    bias_stage = _build_stage(bias_expr, (i_var, j_var), (m_dim, n_dim), outer_assign_map, outer_order_map, "bias")

  return _ScoreContraction(
    stmt_idx=idx,
    stmt=s,
    out_buf=store.buf,
    elt=elt,
    batch_shape=batch_shape,
    batch_index=batch_index,
    m_dim=m_dim,
    n_dim=n_dim,
    k_dim=k_dim,
    i_var=i_var,
    j_var=j_var,
    q_stage=q_stage,
    k_stage=k_stage,
    score_op=score_op,
    score_scale=score_scale,
    mask_cond=mask_cond,
    mask_buf=mask_buf,
    mask_index=mask_index,
    mask_fill=mask_fill,
    pre_checks=pre_checks,
    bias_stage=bias_stage,
    ipa_xq_stage=ipa_xq_stage,
    ipa_xk_stage=ipa_xk_stage,
    ipa_c_dim=ipa_c_dim,
    ipa_gamma=ipa_gamma,
  )


def _unwrap_nonempty_if(stmts: list[Stmt], n_dim: Expr, facts: IndexFacts) -> list[Stmt]:
  """Unwrap ``if (N > 0): body`` wrapper emitted by ``max``/``exp`` fusion."""
  clean = [t for t in stmts if not isinstance(t, (ir.Comment, ReleaseScope))]
  if len(clean) == 1 and isinstance(clean[0], If) and not clean[0].orelse:
    c = clean[0].cond
    if (
      isinstance(c, Prim)
      and c.op == "gt"
      and len(c.args) == 2
      and facts.same_expr(c.args[0], n_dim)
      and isinstance(c.args[1], Const)
      and c.args[1].value == 0
    ):
      return [t for t in clean[0].then if not isinstance(t, (ir.Comment, ReleaseScope))]
  return clean


def _match_row_max_transform_map(
  s: Stmt,
  in_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
  want_exp: bool,
) -> tuple[Var, Var | None, Check | None] | None:
  """Match ``IndexMap(*batch, M)`` doing:
  ``m = reduce(max, in_buf[b.., i, j]); [m_buf[b..,i] = m]; map j: out_buf[b..,i,j] = exp(in_buf - m)``
  (when ``want_exp=True``) or ``out_buf[b..,i,j] = in_buf - m`` (when ``want_exp=False``).
  Returns ``(out_buf, m_buf_or_None, max_check_or_None)``."""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) != len(batch_shape) + 1 or s.scope.owned:
    return None
  if not facts.same_shape(s.shape, (*batch_shape, m_dim)):
    return None
  body = _unwrap_nonempty_if(s.body, n_dim, facts)
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  maps = [k for k, t in enumerate(body) if isinstance(t, IndexMap)]
  if len(reds) != 1 or len(maps) != 1 or reds[0] >= maps[0]:
    return None
  red = body[reds[0]]
  inner_map = body[maps[0]]
  assert isinstance(red, IndexReduce) and isinstance(inner_map, IndexMap)
  if len(red.shape) != 1 or not facts.same_expr(red.shape[0], n_dim) or len(red.folds) != 1:
    return None
  f = red.folds[0]
  if f.op != "max" or f.init is not None or f.combine or f.carries:
    return None
  if len(inner_map.shape) != 1 or not facts.same_expr(inner_map.shape[0], n_dim):
    return None

  # Verify red reads in_buf[b.., i, j]
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body[: reds[0]] + red.body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  fv = _resolve_local(f.value, vals)
  expected_r = row_major_index((*s.index, red.index[0]), (*batch_shape, m_dim, n_dim))
  if not (isinstance(fv, Load) and fv.buf is in_buf and f_facts.same_expr(fv.index, expected_r)):
    return None

  # Optional store of m_buf between red and inner_map (or after inner_map)
  m_buf: Var | None = None
  m_idx = row_major_index(s.index, (*batch_shape, m_dim))
  for t in body:
    if t is red or t is inner_map or isinstance(t, (Assign, Check, ir.Comment)):
      continue
    if isinstance(t, Store) and t.atomic is None and t.value is f.result and f_facts.same_expr(t.index, m_idx):
      if m_buf is not None:
        return None
      m_buf = t.buf
    else:
      return None

  # Verify inner_map stores out_buf[b.., i, j] = exp(in_buf - m) or (in_buf - m)
  im_body = [t for t in inner_map.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  im_stores = [t for t in im_body if isinstance(t, Store)]
  if len(im_stores) != 1 or any(not isinstance(t, (Assign, Store)) for t in im_body):
    return None
  im_st = im_stores[0]
  if im_st.atomic is not None:
    return None
  im_facts = f_facts.fork()
  im_vals = dict(vals)
  for t in im_body:
    im_facts.record_stmt(t)
    if isinstance(t, Assign):
      im_vals[t.var.uid] = t.value
  expected_m = row_major_index((*s.index, inner_map.index[0]), (*batch_shape, m_dim, n_dim))
  if not im_facts.same_expr(im_st.index, expected_m):
    return None
  sv = _resolve_local(im_st.value, im_vals)
  if want_exp:
    if not (isinstance(sv, Prim) and sv.op == "exp" and len(sv.args) == 1):
      return None
    sub_e = sv.args[0]
  else:
    sub_e = sv
  if not (isinstance(sub_e, Prim) and sub_e.op == "sub" and len(sub_e.args) == 2):
    return None
  ld, rhs = sub_e.args
  if rhs is not f.result or not (
    isinstance(ld, Load) and ld.buf is in_buf and im_facts.same_expr(ld.index, expected_m)
  ):
    return None
  max_chk = next((t for t in body if isinstance(t, Check)), None)
  return im_st.buf, m_buf, max_chk


def _match_row_sum_div_map(
  s: Stmt,
  e_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
) -> tuple[Var, Var | None] | None:
  """Match ``IndexMap(*batch, M)`` doing:
  ``l = reduce(add, e_buf[b.., i, j]); [l_buf[b..,i] = l]; map j: p_buf[b..,i,j] = e_buf / l``."""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) != len(batch_shape) + 1 or s.scope.owned:
    return None
  if not facts.same_shape(s.shape, (*batch_shape, m_dim)):
    return None
  body = _unwrap_nonempty_if(s.body, n_dim, facts)
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  maps = [k for k, t in enumerate(body) if isinstance(t, IndexMap)]
  if len(reds) != 1 or len(maps) != 1 or reds[0] >= maps[0]:
    return None
  red = body[reds[0]]
  inner_map = body[maps[0]]
  assert isinstance(red, IndexReduce) and isinstance(inner_map, IndexMap)
  if len(red.shape) != 1 or not facts.same_expr(red.shape[0], n_dim) or len(red.folds) != 1:
    return None
  f = red.folds[0]
  if f.op != "add" or f.combine or f.carries or not (isinstance(f.init, Const) and float(f.init.value) == 0.0):
    return None
  if len(inner_map.shape) != 1 or not facts.same_expr(inner_map.shape[0], n_dim):
    return None

  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body[: reds[0]] + red.body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  fv = _resolve_local(f.value, vals)
  expected_r = row_major_index((*s.index, red.index[0]), (*batch_shape, m_dim, n_dim))
  if not (isinstance(fv, Load) and fv.buf is e_buf and f_facts.same_expr(fv.index, expected_r)):
    return None

  l_buf: Var | None = None
  l_idx = row_major_index(s.index, (*batch_shape, m_dim))
  for t in body:
    if t is red or t is inner_map or isinstance(t, (Assign, ir.Comment)):
      continue
    if isinstance(t, Store) and t.atomic is None and t.value is f.result and f_facts.same_expr(t.index, l_idx):
      if l_buf is not None:
        return None
      l_buf = t.buf
    else:
      return None

  im_body = [t for t in inner_map.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  im_stores = [t for t in im_body if isinstance(t, Store)]
  if len(im_stores) != 1 or any(not isinstance(t, (Assign, Store)) for t in im_body):
    return None
  im_st = im_stores[0]
  if im_st.atomic is not None:
    return None
  im_facts = f_facts.fork()
  im_vals = dict(vals)
  for t in im_body:
    im_facts.record_stmt(t)
    if isinstance(t, Assign):
      im_vals[t.var.uid] = t.value
  expected_m = row_major_index((*s.index, inner_map.index[0]), (*batch_shape, m_dim, n_dim))
  if not im_facts.same_expr(im_st.index, expected_m):
    return None
  sv = _resolve_local(im_st.value, im_vals)
  if not (isinstance(sv, Prim) and sv.op == "div" and len(sv.args) == 2):
    return None
  ld, rhs = sv.args
  if rhs is not f.result or not (isinstance(ld, Load) and ld.buf is e_buf and im_facts.same_expr(ld.index, expected_m)):
    return None
  return im_st.buf, l_buf


def _extract_buf_score_mod(
  e: Expr,
  in_buf: Var,
  expected_idx: Expr,
  facts: IndexFacts,
  inner_uids: set[int],
) -> tuple[bool, Expr | None, Expr | None, Var | None, Expr | None, Expr | None, Expr | None]:
  """Match ``e == [select(mask, ] in_buf[expected_idx] [* scale | / denom | neg] [+ bias | - bias] [, fill)]``.
  Returns ``(ok, scale, mask_cond, mask_buf, mask_index, mask_fill, bias_expr)``."""
  mask_cond: Expr | None = None
  mask_buf: Var | None = None
  mask_index: Expr | None = None
  mask_fill: Expr | None = None
  bias_expr: Expr | None = None
  core = e
  if isinstance(e, Prim) and e.op == "select" and len(e.args) == 3:
    cond_e, then_e, else_e = e.args
    if not _expr_uses_uids(else_e, inner_uids):
      core = then_e
      mask_fill = else_e
      if isinstance(cond_e, Load) and cond_e.buf.type.elt is Bool:
        mask_buf = cond_e.buf
        mask_index = cond_e.index
      elif is_pure_expr(cond_e) and not any(isinstance(x, Load) for x in ir.walk_expr(cond_e)):
        mask_cond = cond_e
      else:
        return (False, None, None, None, None, None, None)
    else:
      return (False, None, None, None, None, None, None)

  # Optional additive/subtractive pair bias around the contraction load
  if isinstance(core, Prim) and core.op in ("add", "sub") and len(core.args) == 2:
    op_ab = core.op
    a, b = core.args
    ua = _expr_uses_uids(a, {in_buf.uid})
    ub = _expr_uses_uids(b, {in_buf.uid})
    if ua and not ub and is_pure_expr(b):
      core = a
      bias_expr = b if op_ab == "add" else ops.prim("neg", (b,), core.type)
    elif op_ab == "add" and ub and not ua and is_pure_expr(a):
      core = b
      bias_expr = a

  scale: Expr | None = None
  if isinstance(core, Load) and core.buf is in_buf and facts.same_expr(core.index, expected_idx):
    return (True, None, mask_cond, mask_buf, mask_index, mask_fill, bias_expr)
  if isinstance(core, Prim) and core.op == "mul" and len(core.args) == 2:
    a, b = core.args
    if (
      isinstance(a, Load)
      and a.buf is in_buf
      and facts.same_expr(a.index, expected_idx)
      and not _expr_uses_uids(b, inner_uids)
    ):
      scale = b
      return (True, scale, mask_cond, mask_buf, mask_index, mask_fill, bias_expr)
    if (
      isinstance(b, Load)
      and b.buf is in_buf
      and facts.same_expr(b.index, expected_idx)
      and not _expr_uses_uids(a, inner_uids)
    ):
      scale = a
      return (True, scale, mask_cond, mask_buf, mask_index, mask_fill, bias_expr)
  if isinstance(core, Prim) and core.op == "div" and len(core.args) == 2:
    a, b = core.args
    if (
      isinstance(a, Load)
      and a.buf is in_buf
      and facts.same_expr(a.index, expected_idx)
      and not _expr_uses_uids(b, inner_uids)
    ):
      scale = ops.prim("div", (Const(1.0, core.type), b), core.type)
      return (True, scale, mask_cond, mask_buf, mask_index, mask_fill, bias_expr)
  if isinstance(core, Prim) and core.op == "neg" and len(core.args) == 1:
    a = core.args[0]
    if isinstance(a, Load) and a.buf is in_buf and facts.same_expr(a.index, expected_idx):
      scale = Const(-1.0, core.type)
      return (True, scale, mask_cond, mask_buf, mask_index, mask_fill, bias_expr)
  return (False, None, None, None, None, None, None)


def _match_row_max_score_mod_map(
  s: Stmt,
  in_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
):
  """Match ``m_buf[b.., i] = reduce_j(max, score_mod(in_buf[b.., i, j]))`` (training form
  where ``scale`` + ``where(mask, ...)`` fused into the ``max`` reduction rather than ``S_map``).
  Returns ``(m_buf, scale, mask_cond, mask_buf, mask_index, mask_fill, row_idx, j_var)`` or ``None``."""
  if not isinstance(s, IndexMap) or s.ordered or s.scope.owned:
    return None
  if facts.same_shape(s.shape, (*batch_shape, m_dim)):
    row_idx = s.index
  elif (
    len(s.shape) == len(batch_shape) + 2
    and facts.same_shape(s.shape[:-1], (*batch_shape, m_dim))
    and isinstance(s.shape[-1], Const)
    and s.shape[-1].value == 1
  ):
    row_idx = s.index[:-1]
  else:
    return None
  body = _unwrap_nonempty_if(s.body, n_dim, facts)
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  stores = [t for t in body if isinstance(t, Store)]
  if len(reds) != 1 or len(stores) != 1:
    return None
  red = body[reds[0]]
  st = stores[0]
  assert isinstance(red, IndexReduce)
  if len(red.shape) != 1 or not facts.same_expr(red.shape[0], n_dim) or len(red.folds) != 1 or st.atomic is not None:
    return None
  f = red.folds[0]
  if f.op != "max" or f.init is not None or f.combine or f.carries:
    return None
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body + red.body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  fv = _resolve_local(f.value, vals)
  expected_r = row_major_index((*row_idx, red.index[0]), (*batch_shape, m_dim, n_dim))
  inner_uids = {v.uid for v in (*s.index, *red.index)} | set(vals)
  ok, scale, mc, mb, mi, mf, bias_e = _extract_buf_score_mod(fv, in_buf, expected_r, f_facts, inner_uids)
  if not ok:
    return None
  row_flat = row_major_index(row_idx, (*batch_shape, m_dim))
  if _resolve_local(st.value, vals) is not f.result or not f_facts.same_expr(st.index, row_flat):
    return None
  return st.buf, scale, mc, mb, mi, mf, bias_e, row_idx, red.index[0]


def _match_row_sum_exp_score_mod_map(
  s: Stmt,
  in_buf: Var,
  m_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
) -> Var | None:
  """Match ``l_buf[b.., i] = reduce_j(add, exp(score_mod(in_buf[b.., i, j]) - m_buf[b.., i]))``."""
  if not isinstance(s, IndexMap) or s.ordered or s.scope.owned:
    return None
  if facts.same_shape(s.shape, (*batch_shape, m_dim)):
    row_idx = s.index
  elif (
    len(s.shape) == len(batch_shape) + 2
    and facts.same_shape(s.shape[:-1], (*batch_shape, m_dim))
    and isinstance(s.shape[-1], Const)
    and s.shape[-1].value == 1
  ):
    row_idx = s.index[:-1]
  else:
    return None
  body = _unwrap_nonempty_if(s.body, n_dim, facts)
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  stores = [t for t in body if isinstance(t, Store)]
  if len(reds) != 1 or len(stores) != 1:
    return None
  red = body[reds[0]]
  st = stores[0]
  assert isinstance(red, IndexReduce)
  if len(red.shape) != 1 or not facts.same_expr(red.shape[0], n_dim) or len(red.folds) != 1 or st.atomic is not None:
    return None
  f = red.folds[0]
  if f.op != "add" or f.combine or f.carries or not (isinstance(f.init, Const) and float(f.init.value) == 0.0):
    return None
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body + red.body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  fv = _resolve_local(f.value, vals)
  if not (isinstance(fv, Prim) and fv.op == "exp" and len(fv.args) == 1):
    return None
  sub_e = fv.args[0]
  if not (isinstance(sub_e, Prim) and sub_e.op == "sub" and len(sub_e.args) == 2):
    return None
  lhs, rhs = sub_e.args
  row_flat = row_major_index(row_idx, (*batch_shape, m_dim))
  if not (isinstance(rhs, Load) and rhs.buf is m_buf and f_facts.same_expr(rhs.index, row_flat)):
    return None
  expected_r = row_major_index((*row_idx, red.index[0]), (*batch_shape, m_dim, n_dim))
  inner_uids = {v.uid for v in (*s.index, *red.index)} | set(vals)
  ok, *_ = _extract_buf_score_mod(lhs, in_buf, expected_r, f_facts, inner_uids)
  if not ok:
    return None
  if _resolve_local(st.value, vals) is not f.result or not f_facts.same_expr(st.index, row_flat):
    return None
  return st.buf


def _match_ew_softmax_prob_map(
  s: Stmt,
  in_buf: Var,
  m_buf: Var,
  l_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
) -> Var | None:
  """Match ``p_buf[b.., i, j] = exp(score_mod(in_buf[b.., i, j]) - m_buf[b.., i]) / l_buf[b.., i]``."""
  if not isinstance(s, IndexMap) or s.ordered or s.scope.owned:
    return None
  if not facts.same_shape(s.shape, (*batch_shape, m_dim, n_dim)):
    return None
  body = [t for t in s.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  stores = [t for t in body if isinstance(t, Store)]
  if len(stores) != 1 or any(not isinstance(t, (Assign, Store)) for t in body):
    return None
  st = stores[0]
  if st.atomic is not None:
    return None
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  full_idx = row_major_index(s.index, (*batch_shape, m_dim, n_dim))
  row_idx = row_major_index(s.index[:-1], (*batch_shape, m_dim))
  if not f_facts.same_expr(st.index, full_idx):
    return None
  sv = _resolve_local(st.value, vals)
  if not (isinstance(sv, Prim) and sv.op == "div" and len(sv.args) == 2):
    return None
  num, den = sv.args
  if not (isinstance(den, Load) and den.buf is l_buf and f_facts.same_expr(den.index, row_idx)):
    return None
  if not (isinstance(num, Prim) and num.op == "exp" and len(num.args) == 1):
    return None
  sub_e = num.args[0]
  if not (isinstance(sub_e, Prim) and sub_e.op == "sub" and len(sub_e.args) == 2):
    return None
  lhs, rhs = sub_e.args
  if not (isinstance(rhs, Load) and rhs.buf is m_buf and f_facts.same_expr(rhs.index, row_idx)):
    return None
  inner_uids = {v.uid for v in s.index} | set(vals)
  ok, *_ = _extract_buf_score_mod(lhs, in_buf, full_idx, f_facts, inner_uids)
  return st.buf if ok else None


def _match_row_reduce_map(
  s: Stmt,
  in_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
  want_op: str,
  want_exp_input: bool = False,
) -> Var | None:
  """Match a standalone row reduction ``IndexMap(*batch, M)`` (or with trailing 1-dim
  ``keepdims=True``) storing ``out_buf[b.., i] = reduce(want_op, [exp(]in_buf[b.., i, j][)])``."""
  if not isinstance(s, IndexMap) or s.ordered or s.scope.owned:
    return None
  # Allow keepdims trailing 1-axis (squeezed or unsqueezed)
  if facts.same_shape(s.shape, (*batch_shape, m_dim)):
    row_idx = s.index
  elif (
    len(s.shape) == len(batch_shape) + 2
    and facts.same_shape(s.shape[:-1], (*batch_shape, m_dim))
    and isinstance(s.shape[-1], Const)
    and s.shape[-1].value == 1
  ):
    row_idx = s.index[:-1]
  else:
    return None
  body = _unwrap_nonempty_if(s.body, n_dim, facts)
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  stores = [t for t in body if isinstance(t, Store)]
  if len(reds) != 1 or len(stores) != 1:
    return None
  red = body[reds[0]]
  st = stores[0]
  assert isinstance(red, IndexReduce)
  if len(red.shape) != 1 or not facts.same_expr(red.shape[0], n_dim) or len(red.folds) != 1 or st.atomic is not None:
    return None
  f = red.folds[0]
  if f.op != want_op or f.combine or f.carries:
    return None
  if want_op == "max" and f.init is not None:
    return None
  if want_op == "add" and not (isinstance(f.init, Const) and float(f.init.value) == 0.0):
    return None
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body + red.body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  fv = _resolve_local(f.value, vals)
  expected_r = row_major_index((*row_idx, red.index[0]), (*batch_shape, m_dim, n_dim))
  if want_exp_input:
    if not (isinstance(fv, Prim) and fv.op == "exp" and len(fv.args) == 1):
      return None
    fv = fv.args[0]
  if not (isinstance(fv, Load) and fv.buf is in_buf and f_facts.same_expr(fv.index, expected_r)):
    return None
  row_flat = row_major_index(row_idx, (*batch_shape, m_dim))
  if _resolve_local(st.value, vals) is not f.result or not f_facts.same_expr(st.index, row_flat):
    return None
  return st.buf


def _match_ew_row_transform_map(
  s: Stmt,
  in_buf: Var,
  stat_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
  kind: str,  # "exp_sub" | "sub" | "div"
) -> Var | None:
  """Match ``IndexMap(*batch, M, N)`` storing ``out[b..,i,j] = exp(in - stat)`` / ``in - stat`` / ``in / stat``."""
  if not isinstance(s, IndexMap) or s.ordered or s.scope.owned:
    return None
  if not facts.same_shape(s.shape, (*batch_shape, m_dim, n_dim)):
    return None
  body = [t for t in s.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  stores = [t for t in body if isinstance(t, Store)]
  if len(stores) != 1 or any(not isinstance(t, (Assign, Store)) for t in body):
    return None
  st = stores[0]
  if st.atomic is not None:
    return None
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  full_idx = row_major_index(s.index, (*batch_shape, m_dim, n_dim))
  row_idx = row_major_index(s.index[:-1], (*batch_shape, m_dim))
  if not f_facts.same_expr(st.index, full_idx):
    return None
  sv = _resolve_local(st.value, vals)
  if kind == "exp_sub":
    if not (isinstance(sv, Prim) and sv.op == "exp" and len(sv.args) == 1):
      return None
    sv = sv.args[0]
    want_bin = "sub"
  elif kind == "sub":
    want_bin = "sub"
  else:
    want_bin = "div"
  if not (isinstance(sv, Prim) and sv.op == want_bin and len(sv.args) == 2):
    return None
  a, b = sv.args
  if not (isinstance(a, Load) and a.buf is in_buf and f_facts.same_expr(a.index, full_idx)):
    return None
  if not (isinstance(b, Load) and b.buf is stat_buf and f_facts.same_expr(b.index, row_idx)):
    return None
  return st.buf


def _match_cce_picked_map(
  s: Stmt,
  z_buf: Var,
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
  allow_dim_eq_check: bool = False,
) -> tuple[Var, list[Stmt], Expr] | None:
  """Match ``IndexMap(M,)`` doing ``picked_buf[i] = z_buf[i, idx_j]``."""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) != 1 or s.scope.owned:
    return None
  if not allow_dim_eq_check and not facts.same_shape(s.shape, (m_dim,)):
    return None
  body = [t for t in s.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  stores = [t for t in body if isinstance(t, Store)]
  if len(stores) != 1 or any(not isinstance(t, (Assign, Store, Check)) for t in body):
    return None
  st = stores[0]
  if st.atomic is not None:
    return None
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  if not f_facts.same_expr(st.index, s.index[0]):
    return None
  ov = _resolve_local(st.value, vals)
  if not (isinstance(ov, Load) and ov.buf is z_buf):
    return None
  if isinstance(ov.index, Prim) and ov.index.op == "add" and len(ov.index.args) == 2:
    p0, p1 = ov.index.args
    if isinstance(p0, Prim) and p0.op == "mul" and len(p0.args) == 2 and f_facts.same_expr(p0.args[1], n_dim):
      if _expr_uses_uids(p0.args[0], {s.index[0].uid}):
        prefix = [t for t in body if isinstance(t, (Assign, Check))]
        return st.buf, prefix, p1
  return None


def _match_cce_gather_only_out_map(
  s: Stmt,
  z_buf: Var,
  s_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  facts: IndexFacts,
) -> tuple[Var, list[Stmt], Expr] | None:
  """Match ``IndexMap(M,)`` doing ``out[i] = z_buf[i, idx_j] - log(s_buf[i])``
  when ``s_buf = sum(exp(z_buf))`` was kept in a separate buffer for AD backward."""
  if len(batch_shape) != 0 or not isinstance(s, IndexMap) or s.ordered or len(s.shape) != 1 or s.scope.owned:
    return None
  if not facts.same_shape(s.shape, (m_dim,)):
    return None
  body = [t for t in s.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  stores = [t for t in body if isinstance(t, Store)]
  if len(stores) != 1 or any(not isinstance(t, (Assign, Store, Check)) for t in body):
    return None
  st = stores[0]
  if st.atomic is not None:
    return None
  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  if not f_facts.same_expr(st.index, s.index[0]):
    return None
  ov = _resolve_local(st.value, vals)
  if not (isinstance(ov, Prim) and ov.op == "sub" and len(ov.args) == 2):
    return None
  lhs, rhs = ov.args
  if not (
    isinstance(rhs, Prim)
    and rhs.op == "log"
    and len(rhs.args) == 1
    and isinstance(rhs.args[0], Load)
    and rhs.args[0].buf is s_buf
    and f_facts.same_expr(rhs.args[0].index, s.index[0])
  ):
    return None
  if not (isinstance(lhs, Load) and lhs.buf is z_buf):
    return None
  if isinstance(lhs.index, Prim) and lhs.index.op == "add" and len(lhs.index.args) == 2:
    p0, p1 = lhs.index.args
    if isinstance(p0, Prim) and p0.op == "mul" and len(p0.args) == 2 and f_facts.same_expr(p0.args[1], n_dim):
      if _expr_uses_uids(p0.args[0], {s.index[0].uid}):
        prefix = [t for t in body if isinstance(t, (Assign, Check))]
        return st.buf, prefix, p1
  return None


def _match_value_contraction(
  s: Stmt,
  p_buf: Var,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  elt: ScalarType,
  facts: IndexFacts,
) -> tuple[Var, Expr, TileStage, tuple[Var, ...]] | None:
  """Match ``IndexMap(*batch, M, D)`` doing ``O[b.., i, d] = reduce_j P[b.., i, j] * V[b.., j, d]``.
  Returns ``(out_buf, d_dim, v_stage, o_batch_index)``."""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) != len(batch_shape) + 2 or s.scope.owned:
    return None
  if not facts.same_shape(s.shape[:-1], (*batch_shape, m_dim)):
    return None
  body = [t for t in s.body if not isinstance(t, (ir.Comment, ReleaseScope))]
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  if len(reds) != 1:
    return None
  r_pos = reds[0]
  red = body[r_pos]
  assert isinstance(red, IndexReduce)
  if len(red.shape) != 1 or not facts.same_expr(red.shape[0], n_dim) or len(red.folds) != 1 or red.where is not None:
    return None
  fold = red.folds[0]
  if (
    fold.op != "add"
    or fold.combine
    or fold.carries
    or fold.a.type is not elt
    or not (isinstance(fold.init, Const) and float(fold.init.value) == 0.0)
  ):
    return None
  before = body[:r_pos]
  after = body[r_pos + 1 :]
  if any(not isinstance(t, (Assign, ir.Comment)) for t in before + red.body):
    return None
  stores = [t for t in after if isinstance(t, Store)]
  if len(stores) != 1 or any(not isinstance(t, (Assign, Store, ir.Comment)) for t in after):
    return None
  st = stores[0]
  if st.atomic is not None or st.buf.type.elt is not elt:
    return None

  local_facts = facts.fork()
  all_assigns = [t for t in before + red.body + after if isinstance(t, Assign)]
  assign_map = {t.var.uid: t for t in all_assigns}
  order_map = {t.var.uid: k for k, t in enumerate(all_assigns)}
  vals = {t.var.uid: t.value for t in all_assigns}
  for t in all_assigns:
    local_facts.record_stmt(t)
  expected_o = row_major_index(s.index, s.shape)
  if not local_facts.same_expr(st.index, expected_o) or _resolve_local(st.value, vals) is not fold.result:
    return None

  fv = _resolve_local(fold.value, vals)
  if not (isinstance(fv, Prim) and fv.op == "mul" and len(fv.args) == 2):
    return None
  i_var, d_var = s.index[-2], s.index[-1]
  j_var = red.index[0]
  expected_p = row_major_index((*s.index[:-1], j_var), (*batch_shape, m_dim, n_dim))
  a, b = fv.args
  if isinstance(a, Load) and a.buf is p_buf and local_facts.same_expr(a.index, expected_p):
    v_expr = b
  elif isinstance(b, Load) and b.buf is p_buf and local_facts.same_expr(b.index, expected_p):
    v_expr = a
  else:
    return None

  if not is_pure_expr(v_expr) or _expr_uses_uids(v_expr, {i_var.uid}):
    return None
  d_dim = s.shape[-1]
  v_stage = _build_stage(v_expr, (j_var, d_var), (n_dim, d_dim), assign_map, order_map, "v")
  return st.buf, d_dim, v_stage, s.index[:-2]


def _rebind_stage_batch(st: TileStage, old_batch: tuple[Var, ...], new_batch: tuple[Var, ...]) -> TileStage:
  """Rebind batch index variables inside ``st`` to ``new_batch``."""
  if not old_batch or all(a is b for a, b in zip(old_batch, new_batch)):
    return st
  cl = Cloner({a.uid: b for a, b in zip(old_batch, new_batch)})
  for v in st.index:
    cl.m[v.uid] = v
  for v in defined(st.body).values():
    cl.var(v)
  return TileStage(st.buf, st.shape, st.index, [cl.stmt(t) for t in st.body], cl.expr(st.value), Scope("stage"))


def _inline_stage(st: TileStage, coords: tuple[Expr, ...], out_stmts: list[Stmt]) -> Expr:
  cl = Cloner({v.uid: c for v, c in zip(st.index, coords)})
  for v in defined(st.body).values():
    cl.var(v)
  out_stmts.extend(cl.stmt(t) for t in st.body)
  return cl.expr(st.value)


def _derive_law5_guards(
  mask_cond: Expr | None,
  i_var: Var,
  j_var: Var,
  facts: IndexFacts,
  bm: int = 32,
  bn: int = 32,
) -> tuple[Expr | None, Expr | None, tuple[Var, Var] | None]:
  """Run Law 5 affine tile-interval analysis on symbolic ``mask_cond(i, j)``."""
  if mask_cond is None:
    return None, None, None
  i0 = Var("pk_ti0", I)
  j0 = Var("pk_tj0", I)
  iv_map = {
    i_var.uid: TileInterval(i0, iadd(i0, Const(bm - 1, I))),
    j_var.uid: TileInterval(j0, iadd(j0, Const(bn - 1, I))),
  }
  af, at = tile_predicate_guards(mask_cond, iv_map, facts)
  af_out = None if (isinstance(af, Const) and not af.value) else af
  at_out = None if (isinstance(at, Const) and not at.value) else at
  if af_out is None and at_out is None:
    return None, None, None
  return af_out, at_out, (i0, j0)


def _stage_uses_var(st: TileStage, uid: int) -> bool:
  for s in ir.walk_stmts(st.body):
    for e in ir.stmt_exprs(s):
      if _expr_uses_uids(e, {uid}):
        return True
  return _expr_uses_uids(st.value, {uid})


def _maybe_fold_gqa_group(
  sc: _ScoreContraction,
  v_stage: TileStage,
  v_batch: tuple[Var, ...],
) -> tuple[_ScoreContraction, TileStage, tuple[Var, ...], list[Stmt]]:
  """Law 3 (GQA / Shared-KV Head-Group Tiling):

  When the trailing batch axis ``g`` of ``Q`` (extent ``G``) does not appear in
  ``K`` or ``V`` (e.g. Grouped-Query Attention ``Q: (B, H_kv, G, T_q, D)``,
  ``K, V: (B, H_kv, 1, T_kv, D)``), fold ``g`` into the query row axis::

      i_eff = g * M + i    (extent G * M)

  so a single SRAM load of ``K`` and ``V`` serves all ``G`` query heads
  simultaneously (cutting KV cache HBM traffic by ``G``).  Row-major layouts of
  ``O``, ``m``, and ``l`` are bit-for-bit unchanged.
  """
  if not sc.batch_shape or len(v_batch) != len(sc.batch_index):
    return sc, v_stage, v_batch, []
  if sc.bias_stage is not None or sc.ipa_xq_stage is not None:
    return sc, v_stage, v_batch, []
  # Keep Law 5 causal/sliding-window interval pruning intact when masked with symbolic M > 1
  if sc.mask_cond is not None or sc.mask_buf is not None:
    if not (isinstance(sc.m_dim, Const) and int(sc.m_dim.value) == 1):
      return sc, v_stage, v_batch, []
  g_var = sc.batch_index[-1]
  g_dim = sc.batch_shape[-1]
  vg_var = v_batch[-1]
  if not _stage_uses_var(sc.q_stage, g_var.uid):
    return sc, v_stage, v_batch, []
  if _stage_uses_var(sc.k_stage, g_var.uid) or _stage_uses_var(v_stage, vg_var.uid):
    return sc, v_stage, v_batch, []
  if sc.mask_cond is not None and _expr_uses_uids(sc.mask_cond, {g_var.uid}):
    return sc, v_stage, v_batch, []
  if sc.mask_index is not None and _expr_uses_uids(sc.mask_index, {g_var.uid}):
    return sc, v_stage, v_batch, []

  m_eff_v = Var("gqa_meff", I)
  prelude: list[Stmt] = [Assign(m_eff_v, imul(g_dim, sc.m_dim))]
  i_eff = Var("gqa_ieff", I)
  k_new = Var("gqa_k", I)
  g_loc = Var("gqa_g", I)
  i_loc = Var("gqa_i", I)
  q_cl = Cloner({g_var.uid: g_loc, sc.q_stage.index[0].uid: i_loc, sc.q_stage.index[1].uid: k_new})
  q_body: list[Stmt] = [
    Assign(g_loc, idiv(i_eff, sc.m_dim)),
    Assign(i_loc, imod(i_eff, sc.m_dim)),
    *(q_cl.stmt(t) for t in sc.q_stage.body),
  ]
  q_folded = TileStage(
    buf=Var("gqa_qbuf", sc.q_stage.buf.type),
    shape=(m_eff_v, sc.k_dim),
    index=(i_eff, k_new),
    body=q_body,
    value=q_cl.expr(sc.q_stage.value),
    scope=Scope("stage"),
  )
  # Mask uses i = 0 when M == 1
  mask_cond = sc.mask_cond
  mask_index = sc.mask_index
  if mask_cond is not None:
    mask_cond = Cloner({sc.i_var.uid: ZERO}).expr(mask_cond)
  if mask_index is not None:
    mask_index = Cloner({sc.i_var.uid: ZERO}).expr(mask_index)

  sc_folded = _ScoreContraction(
    stmt_idx=sc.stmt_idx,
    stmt=sc.stmt,
    out_buf=sc.out_buf,
    elt=sc.elt,
    batch_shape=sc.batch_shape[:-1],
    batch_index=sc.batch_index[:-1],
    m_dim=m_eff_v,
    n_dim=sc.n_dim,
    k_dim=sc.k_dim,
    i_var=sc.i_var,
    j_var=sc.j_var,
    q_stage=q_folded,
    k_stage=sc.k_stage,
    score_op=sc.score_op,
    score_scale=sc.score_scale,
    mask_cond=mask_cond,
    mask_buf=sc.mask_buf,
    mask_index=mask_index,
    mask_fill=sc.mask_fill,
    pre_checks=sc.pre_checks,
  )
  return sc_folded, v_stage, v_batch[:-1], prelude


def _build_flash_monoid_tile(
  sc: _ScoreContraction,
  o_buf: Var,
  d_dim: Expr,
  v_stage: TileStage,
  v_batch: tuple[Var, ...],
  m_buf: Var | None,
  l_buf: Var | None,
  lse_buf: Var | None,
  facts: IndexFacts,
) -> IndexTile:
  """Build ONE :class:`IndexTile` with 1-pass rescaled ``(m, l, O)`` monoid
  CPU fallback AND :class:`MonoidalTilePlan` (``kind="flash_monoid"``) for GPU."""
  elt = sc.elt
  shape = (*sc.batch_shape, sc.m_dim)
  tile_shape: tuple[Expr | None, ...] = (ONE,) * len(shape)
  origin = tuple(Var("fo", I) for _ in shape)
  extent = tuple(Var("fe", I) for _ in shape)
  b_vars = origin[:-1]
  i_var = origin[-1]

  # Rebind q, k, v stages (and optional bias / IPA stages + mask expr) onto origin batch vars
  q_st = _rebind_stage_batch(sc.q_stage, sc.batch_index, b_vars)
  k_st = _rebind_stage_batch(sc.k_stage, sc.batch_index, b_vars)
  v_st = _rebind_stage_batch(v_stage, v_batch, b_vars)
  bias_st = None if sc.bias_stage is None else _rebind_stage_batch(sc.bias_stage, sc.batch_index, b_vars)
  xq_st = None if sc.ipa_xq_stage is None else _rebind_stage_batch(sc.ipa_xq_stage, sc.batch_index, b_vars)
  xk_st = None if sc.ipa_xk_stage is None else _rebind_stage_batch(sc.ipa_xk_stage, sc.batch_index, b_vars)

  # Rebind mask_cond / mask_index to fresh (i_m, j_m) and b_vars
  i_m, j_m = Var("mi", I), Var("mj", I)
  m_subst = {a.uid: b for a, b in zip(sc.batch_index, b_vars)}
  m_subst[sc.i_var.uid] = i_m
  m_subst[sc.j_var.uid] = j_m
  m_cl = Cloner(m_subst)
  mask_cond = None if sc.mask_cond is None else m_cl.expr(sc.mask_cond)
  mask_index = None if sc.mask_index is None else m_cl.expr(sc.mask_index)

  tile_false_g, tile_true_g, tile_ov = _derive_law5_guards(mask_cond, i_m, j_m, facts)

  # --- CPU fallback body: 1-pass O(D) rescaled monoid over j in [0, N) ---
  body: list[Stmt] = []
  scope = Scope("loop")
  row_flat = row_major_index((*b_vars, i_var), (*sc.batch_shape, sc.m_dim))
  o_row_base = Var("o_base", I)
  body.append(Assign(o_row_base, imul(row_flat, d_dim)))

  # Zero-init O row
  d0 = Var("d0", I)
  zero_c = Const(0.0, elt)
  one_c = Const(1.0, elt)
  body.append(ForRange(d0, ZERO, d_dim, ONE, [Store(o_buf, iadd(o_row_base, d0), zero_c)], Scope("loop"), simd=True))

  m_acc = Var("fm_acc", elt)
  l_acc = Var("fl_acc", elt)
  body.append(Assign(m_acc, Const(float("-inf"), elt)))
  body.append(Assign(l_acc, zero_c))

  # Loop over j in [0, N)
  j_lp = Var("fj", I)
  j_blk: list[Stmt] = []
  # Optional mask short-circuit on CPU when mask_fill is -inf? No, preserve exact fill semantics (e.g. -1e9)!
  k_lp = Var("fk", I)
  s_acc = Var("fs_acc", elt)
  j_blk.append(Assign(s_acc, zero_c))
  k_blk: list[Stmt] = []
  qv = _inline_stage(q_st, (i_var, k_lp), k_blk)
  kv = _inline_stage(k_st, (j_lp, k_lp), k_blk)
  if sc.score_op == "sqdist":
    diff = Var("fdiff", elt)
    k_blk.append(Assign(diff, ops.prim("sub", (qv, kv), elt)))
    term: Expr = ops.prim("mul", (diff, diff), elt)
  else:
    term = ops.prim("mul", (qv, kv), elt)
  k_blk.append(Assign(s_acc, ops.prim("add", (s_acc, term), elt)))
  j_blk.append(ForRange(k_lp, ZERO, sc.k_dim, ONE, k_blk, Scope("loop"), simd=True, reduction=("add", [s_acc])))

  s_val: Expr = s_acc
  if sc.score_scale is not None:
    s_val = ops.prim("mul", (s_val, cast(sc.score_scale, elt)), elt)
  if xq_st is not None and xk_st is not None and sc.ipa_c_dim is not None and sc.ipa_gamma is not None:
    c_lp = Var("fc", I)
    sq_acc = Var("fsq_acc", elt)
    j_blk.append(Assign(sq_acc, zero_c))
    c_blk: list[Stmt] = []
    xqv = _inline_stage(xq_st, (i_var, c_lp), c_blk)
    xkv = _inline_stage(xk_st, (j_lp, c_lp), c_blk)
    cdf = Var("fcdf", elt)
    c_blk.append(Assign(cdf, ops.prim("sub", (xqv, xkv), elt)))
    c_blk.append(Assign(sq_acc, ops.prim("add", (sq_acc, ops.prim("mul", (cdf, cdf), elt)), elt)))
    j_blk.append(ForRange(c_lp, ZERO, sc.ipa_c_dim, ONE, c_blk, Scope("loop"), simd=True, reduction=("add", [sq_acc])))
    s_val = ops.prim("sub", (s_val, ops.prim("mul", (cast(sc.ipa_gamma, elt), sq_acc), elt)), elt)
  if bias_st is not None:
    bv = _inline_stage(bias_st, (i_var, j_lp), j_blk)
    s_val = ops.prim("add", (s_val, cast(bv, elt)), elt)
  if sc.mask_fill is not None and (mask_cond is not None or sc.mask_buf is not None):
    c_cl = Cloner({i_m.uid: i_var, j_m.uid: j_lp})
    if sc.mask_buf is not None and mask_index is not None:
      cond_at: Expr = Load(sc.mask_buf, c_cl.expr(mask_index))
    else:
      assert mask_cond is not None
      cond_at = c_cl.expr(mask_cond)
    s_val = ops.select(cond_at, s_val, cast(sc.mask_fill, elt))
  s_cur = Var("fs_cur", elt)
  j_blk.append(Assign(s_cur, s_val))

  # Online rescaled monoid update: (m, l, O) (+) (s_cur, 1, V[j])
  m_new = Var("fm_new", elt)
  alpha = Var("falpha", elt)
  beta = Var("fbeta", elt)
  is_first = cmp("eq", j_lp, ZERO)
  j_blk.append(Assign(m_new, ops.select(is_first, s_cur, ops.prim("max", (m_acc, s_cur), elt))))
  j_blk.append(
    Assign(alpha, ops.select(is_first, zero_c, ops.prim("exp", (ops.prim("sub", (m_acc, m_new), elt),), elt)))
  )
  j_blk.append(Assign(beta, ops.select(is_first, one_c, ops.prim("exp", (ops.prim("sub", (s_cur, m_new), elt),), elt))))
  j_blk.append(Assign(l_acc, ops.prim("add", (ops.prim("mul", (l_acc, alpha), elt), beta), elt)))
  j_blk.append(Assign(m_acc, m_new))

  d_lp = Var("fd", I)
  d_blk: list[Stmt] = []
  vv = _inline_stage(v_st, (j_lp, d_lp), d_blk)
  o_pos = iadd(o_row_base, d_lp)
  d_blk.append(
    Store(
      o_buf,
      o_pos,
      ops.prim("add", (ops.prim("mul", (Load(o_buf, o_pos), alpha), elt), ops.prim("mul", (beta, vv), elt)), elt),
    )
  )
  j_blk.append(ForRange(d_lp, ZERO, d_dim, ONE, d_blk, Scope("loop"), simd=True))
  body.append(ForRange(j_lp, ZERO, sc.n_dim, ONE, j_blk, Scope("loop")))

  # Final normalize: O /= l_acc, and optionally store m / l / lse
  inv_l = Var("finv_l", elt)
  body.append(Assign(inv_l, ops.prim("div", (one_c, l_acc), elt)))
  d_end = Var("fd_end", I)
  end_pos = iadd(o_row_base, d_end)
  body.append(
    ForRange(
      d_end,
      ZERO,
      d_dim,
      ONE,
      [Store(o_buf, end_pos, ops.prim("mul", (Load(o_buf, end_pos), inv_l), elt))],
      Scope("loop"),
      simd=True,
    )
  )
  out_bufs: dict[str, Var] = {"o": o_buf}
  if m_buf is not None:
    body.append(Store(m_buf, row_flat, m_acc))
    out_bufs["m"] = m_buf
  if l_buf is not None:
    body.append(Store(l_buf, row_flat, l_acc))
    out_bufs["l"] = l_buf
  if lse_buf is not None:
    body.append(Store(lse_buf, row_flat, ops.prim("add", (m_acc, ops.prim("log", (l_acc,), elt)), elt)))
    out_bufs["lse"] = lse_buf
  body.append(ReleaseScope(scope))

  stages: dict[str, TileStage] = {"q": q_st, "k": k_st, "v": v_st}
  aux_exprs: dict[str, Expr] = {}
  if bias_st is not None:
    stages["bias"] = bias_st
  if xq_st is not None and xk_st is not None and sc.ipa_c_dim is not None and sc.ipa_gamma is not None:
    stages["xq"] = xq_st
    stages["xk"] = xk_st
    aux_exprs["ipa_gamma"] = cast(sc.ipa_gamma, elt)
    aux_exprs["ipa_c_dim"] = sc.ipa_c_dim

  plan = MonoidalTilePlan(
    kind="flash_monoid",
    elt=elt,
    batch_shape=sc.batch_shape,
    batch_index=b_vars,
    m_dim=sc.m_dim,
    n_dim=sc.n_dim,
    k_dim=sc.k_dim,
    d_dim=d_dim,
    stages=stages,
    score_op=sc.score_op,
    score_scale=sc.score_scale,
    mask_cond=mask_cond,
    mask_index_vars=(i_m, j_m),
    mask_buf=sc.mask_buf,
    mask_index=mask_index,
    mask_fill=sc.mask_fill,
    tile_false_guard=tile_false_g,
    tile_true_guard=tile_true_g,
    tile_origin_vars=tile_ov,
    out_bufs=out_bufs,
    aux_exprs=aux_exprs,
    split_k=True,
  )
  return IndexTile(
    shape=shape,
    tile_shape=tile_shape,
    origin=origin,
    extent=extent,
    stages=[],
    body=body,
    scope=scope,
    level="shared",
    parallel=sc.stmt.parallel,
    loc=sc.stmt.loc,
    monoidal_tile=plan,
  )


def _maybe_wrap_flash_decoding(
  sc: _ScoreContraction,
  o_buf: Var,
  d_dim: Expr,
  v_stage: TileStage,
  v_batch: tuple[Var, ...],
  m_buf: Var | None,
  l_buf: Var | None,
  single_tile: IndexTile,
  facts: IndexFacts,
  owner: Scope,
  target,
) -> list[Stmt]:
  """FlashDecoding (Law 1 associativity): when ``target`` is a GPU and the query
  grid is small (e.g. autoregressive decode ``M <= 16``, ``N >= 128``), split
  ``N`` across ``S`` parallel blocks computing partial ``(m_p, l_p, O_p)`` and
  combine them via the exact same rescaled monoid ``(+)``."""
  if target is None or not target.is_gpu or sc.bias_stage is not None or sc.ipa_xq_stage is not None:
    return [single_tile]
  sm_count = max(int(getattr(target, "num_compute_units", 0)), 4)
  elt = sc.elt
  zero_c = Const(0.0, elt)
  one_c = Const(1.0, elt)

  # Guard: M > 0 and M <= 16 and N >= 128 and batch * ceil(M/32) < sm_count
  batch_prod = iprod(sc.batch_shape) if sc.batch_shape else ONE
  cond = ops.prim(
    "and",
    (
      ops.prim(
        "and",
        (cmp("gt", sc.m_dim, ZERO), cmp("le", sc.m_dim, Const(16, I))),
        Bool,
      ),
      ops.prim(
        "and",
        (cmp("ge", sc.n_dim, Const(128, I)), cmp("lt", batch_prod, Const(sm_count, I))),
        Bool,
      ),
    ),
    Bool,
  )

  then_blk: list[Stmt] = []
  sk = Var("fd_sk", I)
  want_sk = idiv(Const(sm_count * 2, I), imax(batch_prod, ONE))
  sk_expr = imin(Const(16, I), imin(idiv(sc.n_dim, Const(32, I)), imax(want_sk, Const(2, I))))
  then_blk.append(Assign(sk, sk_expr))

  # Allocate partial buffers: m_part, l_part (batch * sk * M), o_part (batch * sk * M * D)
  part_rows = Var("fd_prows", I)
  then_blk.append(Assign(part_rows, imul(imul(batch_prod, sk), sc.m_dim)))
  m_part = Var("fd_mp", BufType(elt))
  l_part = Var("fd_lp", BufType(elt))
  o_part = Var("fd_op", BufType(elt))
  owner.owned.extend([m_part, l_part, o_part])
  then_blk.append(ir.Alloc(m_part, part_rows, zero=False))
  then_blk.append(ir.Alloc(l_part, part_rows, zero=False))
  then_blk.append(ir.Alloc(o_part, imul(part_rows, d_dim), zero=False))

  # Pass 1: IndexTile over (*batch, sk, M) with j in [p*N/sk, (p+1)*N/sk)
  p1_shape = (*sc.batch_shape, sk, sc.m_dim)
  p1_origin = tuple(Var("fdo", I) for _ in p1_shape)
  p1_extent = tuple(Var("fde", I) for _ in p1_shape)
  b_vars = p1_origin[:-2]
  p_var = p1_origin[-2]
  i_var = p1_origin[-1]
  j_start_e = idiv(imul(p_var, sc.n_dim), sk)
  j_end_e = idiv(imul(iadd(p_var, ONE), sc.n_dim), sk)

  q_st = _rebind_stage_batch(sc.q_stage, sc.batch_index, b_vars)
  k_st = _rebind_stage_batch(sc.k_stage, sc.batch_index, b_vars)
  v_st = _rebind_stage_batch(v_stage, v_batch, b_vars)
  bias_st = None if sc.bias_stage is None else _rebind_stage_batch(sc.bias_stage, sc.batch_index, b_vars)
  xq_st = None if sc.ipa_xq_stage is None else _rebind_stage_batch(sc.ipa_xq_stage, sc.batch_index, b_vars)
  xk_st = None if sc.ipa_xk_stage is None else _rebind_stage_batch(sc.ipa_xk_stage, sc.batch_index, b_vars)

  i_m, j_m = Var("fdmi", I), Var("fdmj", I)
  m_subst = {a.uid: b for a, b in zip(sc.batch_index, b_vars)}
  m_subst[sc.i_var.uid] = i_m
  m_subst[sc.j_var.uid] = j_m
  m_cl = Cloner(m_subst)
  mask_cond = None if sc.mask_cond is None else m_cl.expr(sc.mask_cond)
  mask_index = None if sc.mask_index is None else m_cl.expr(sc.mask_index)
  tile_false_g, tile_true_g, tile_ov = _derive_law5_guards(mask_cond, i_m, j_m, facts)

  # Pass 1 CPU fallback body (unnormalized partial (m_p, l_p, O_p) over [j_start, j_end))
  p1_body: list[Stmt] = []
  p1_scope = Scope("loop")
  row_flat = row_major_index((*b_vars, p_var, i_var), p1_shape)
  o_row_base = Var("fdo_b", I)
  p1_body.append(Assign(o_row_base, imul(row_flat, d_dim)))
  d0 = Var("fdd0", I)
  p1_body.append(
    ForRange(d0, ZERO, d_dim, ONE, [Store(o_part, iadd(o_row_base, d0), zero_c)], Scope("loop"), simd=True)
  )
  m_acc = Var("fdm_a", elt)
  l_acc = Var("fdl_a", elt)
  p1_body.append(Assign(m_acc, Const(float("-inf"), elt)))
  p1_body.append(Assign(l_acc, zero_c))
  js_v = Var("fd_js", I)
  je_v = Var("fd_je", I)
  p1_body.append(Assign(js_v, j_start_e))
  p1_body.append(Assign(je_v, j_end_e))

  j_lp = Var("fdj", I)
  j_blk: list[Stmt] = []
  k_lp = Var("fdk", I)
  s_acc = Var("fds_a", elt)
  j_blk.append(Assign(s_acc, zero_c))
  k_blk: list[Stmt] = []
  qv = _inline_stage(q_st, (i_var, k_lp), k_blk)
  kv = _inline_stage(k_st, (j_lp, k_lp), k_blk)
  if sc.score_op == "sqdist":
    df = Var("fddf", elt)
    k_blk.append(Assign(df, ops.prim("sub", (qv, kv), elt)))
    term: Expr = ops.prim("mul", (df, df), elt)
  else:
    term = ops.prim("mul", (qv, kv), elt)
  k_blk.append(Assign(s_acc, ops.prim("add", (s_acc, term), elt)))
  j_blk.append(ForRange(k_lp, ZERO, sc.k_dim, ONE, k_blk, Scope("loop"), simd=True, reduction=("add", [s_acc])))
  s_val: Expr = s_acc
  if sc.score_scale is not None:
    s_val = ops.prim("mul", (s_val, cast(sc.score_scale, elt)), elt)
  if xq_st is not None and xk_st is not None and sc.ipa_c_dim is not None and sc.ipa_gamma is not None:
    c_lp = Var("fc", I)
    sq_acc = Var("fsq_acc", elt)
    j_blk.append(Assign(sq_acc, zero_c))
    c_blk: list[Stmt] = []
    xqv = _inline_stage(xq_st, (i_var, c_lp), c_blk)
    xkv = _inline_stage(xk_st, (j_lp, c_lp), c_blk)
    cdf = Var("fcdf", elt)
    c_blk.append(Assign(cdf, ops.prim("sub", (xqv, xkv), elt)))
    c_blk.append(Assign(sq_acc, ops.prim("add", (sq_acc, ops.prim("mul", (cdf, cdf), elt)), elt)))
    j_blk.append(ForRange(c_lp, ZERO, sc.ipa_c_dim, ONE, c_blk, Scope("loop"), simd=True, reduction=("add", [sq_acc])))
    s_val = ops.prim("sub", (s_val, ops.prim("mul", (cast(sc.ipa_gamma, elt), sq_acc), elt)), elt)
  if bias_st is not None:
    bv = _inline_stage(bias_st, (i_var, j_lp), j_blk)
    s_val = ops.prim("add", (s_val, cast(bv, elt)), elt)
  if sc.mask_fill is not None and (mask_cond is not None or sc.mask_buf is not None):
    c_cl = Cloner({i_m.uid: i_var, j_m.uid: j_lp})
    cond_at: Expr = (
      Load(sc.mask_buf, c_cl.expr(mask_index))
      if (sc.mask_buf is not None and mask_index is not None)
      else c_cl.expr(mask_cond)
    )  # type: ignore[arg-type]
    s_val = ops.select(cond_at, s_val, cast(sc.mask_fill, elt))
  s_cur = Var("fds_c", elt)
  j_blk.append(Assign(s_cur, s_val))
  m_new = Var("fdm_n", elt)
  alpha = Var("fdal", elt)
  beta = Var("fdbe", elt)
  is_first = cmp("eq", j_lp, js_v)
  j_blk.append(Assign(m_new, ops.select(is_first, s_cur, ops.prim("max", (m_acc, s_cur), elt))))
  j_blk.append(
    Assign(alpha, ops.select(is_first, zero_c, ops.prim("exp", (ops.prim("sub", (m_acc, m_new), elt),), elt)))
  )
  j_blk.append(Assign(beta, ops.select(is_first, one_c, ops.prim("exp", (ops.prim("sub", (s_cur, m_new), elt),), elt))))
  j_blk.append(Assign(l_acc, ops.prim("add", (ops.prim("mul", (l_acc, alpha), elt), beta), elt)))
  j_blk.append(Assign(m_acc, m_new))
  d_lp = Var("fdd", I)
  d_blk: list[Stmt] = []
  vv = _inline_stage(v_st, (j_lp, d_lp), d_blk)
  op_pos = iadd(o_row_base, d_lp)
  d_blk.append(
    Store(
      o_part,
      op_pos,
      ops.prim("add", (ops.prim("mul", (Load(o_part, op_pos), alpha), elt), ops.prim("mul", (beta, vv), elt)), elt),
    )
  )
  j_blk.append(ForRange(d_lp, ZERO, d_dim, ONE, d_blk, Scope("loop"), simd=True))
  p1_body.append(ForRange(j_lp, js_v, je_v, ONE, j_blk, Scope("loop")))
  p1_body.append(Store(m_part, row_flat, m_acc))
  p1_body.append(Store(l_part, row_flat, l_acc))
  p1_body.append(ReleaseScope(p1_scope))

  p1_plan = MonoidalTilePlan(
    kind="flash_monoid",
    elt=elt,
    batch_shape=(*sc.batch_shape, sk),
    batch_index=(*b_vars, p_var),
    m_dim=sc.m_dim,
    n_dim=sc.n_dim,
    k_dim=sc.k_dim,
    d_dim=d_dim,
    stages={"q": q_st, "k": k_st, "v": v_st},
    score_op=sc.score_op,
    score_scale=sc.score_scale,
    mask_cond=mask_cond,
    mask_index_vars=(i_m, j_m),
    mask_buf=sc.mask_buf,
    mask_index=mask_index,
    mask_fill=sc.mask_fill,
    tile_false_guard=tile_false_g,
    tile_true_guard=tile_true_g,
    tile_origin_vars=tile_ov,
    out_bufs={"o": o_part, "m": m_part, "l": l_part},
    aux_exprs={"j_start": j_start_e, "j_end": j_end_e, "unnormalized": Const(1, I)},
    split_k=True,
  )
  then_blk.append(
    IndexTile(
      shape=p1_shape,
      tile_shape=(ONE,) * len(p1_shape),
      origin=p1_origin,
      extent=p1_extent,
      stages=[],
      body=p1_body,
      scope=p1_scope,
      level="shared",
      parallel=True,
      loc=sc.stmt.loc,
      monoidal_tile=p1_plan,
    )
  )

  # Pass 2: combine S partial (m_p, l_p, O_p) via the exact same associative rescaled monoid (+)
  c_shape = (*sc.batch_shape, sc.m_dim)
  c_idx = tuple(Var("fdc", I) for _ in c_shape)
  c_body: list[Stmt] = []
  c_scope = Scope("loop")
  cb_vars, ci_var = c_idx[:-1], c_idx[-1]
  out_row = row_major_index(c_idx, c_shape)
  out_base = Var("fdob", I)
  c_body.append(Assign(out_base, imul(out_row, d_dim)))

  # First compute global m_tot = max_p m_p and l_tot = sum_p l_p * exp(m_p - m_tot)
  m_tot = Var("fd_mt", elt)
  l_tot = Var("fd_lt", elt)
  p0_row = row_major_index((*cb_vars, ZERO, ci_var), p1_shape)
  c_body.append(Assign(m_tot, Load(m_part, p0_row)))
  p_lp1 = Var("fdp1", I)
  c_body.append(
    ForRange(
      p_lp1,
      ONE,
      sk,
      ONE,
      [
        Assign(m_tot, ops.prim("max", (m_tot, Load(m_part, row_major_index((*cb_vars, p_lp1, ci_var), p1_shape))), elt))
      ],
      Scope("loop"),
    )
  )
  c_body.append(Assign(l_tot, zero_c))
  p_lp2 = Var("fdp2", I)
  pr2 = row_major_index((*cb_vars, p_lp2, ci_var), p1_shape)
  c_body.append(
    ForRange(
      p_lp2,
      ZERO,
      sk,
      ONE,
      [
        Assign(
          l_tot,
          ops.prim(
            "add",
            (
              l_tot,
              ops.prim(
                "mul",
                (Load(l_part, pr2), ops.prim("exp", (ops.prim("sub", (Load(m_part, pr2), m_tot), elt),), elt)),
                elt,
              ),
            ),
            elt,
          ),
        )
      ],
      Scope("loop"),
    )
  )
  inv_lt = Var("fd_invl", elt)
  c_body.append(Assign(inv_lt, ops.prim("div", (one_c, l_tot), elt)))

  # Combine O[d] = (sum_p O_p[d] * exp(m_p - m_tot)) / l_tot
  cd_lp = Var("fdcd", I)
  cd_blk: list[Stmt] = []
  o_sum = Var("fdos", elt)
  cd_blk.append(Assign(o_sum, zero_c))
  p_lp3 = Var("fdp3", I)
  pr3 = row_major_index((*cb_vars, p_lp3, ci_var), p1_shape)
  w_p = ops.prim("exp", (ops.prim("sub", (Load(m_part, pr3), m_tot), elt),), elt)
  cd_blk.append(
    ForRange(
      p_lp3,
      ZERO,
      sk,
      ONE,
      [
        Assign(
          o_sum, ops.prim("add", (o_sum, ops.prim("mul", (Load(o_part, iadd(imul(pr3, d_dim), cd_lp)), w_p), elt)), elt)
        )
      ],
      Scope("loop"),
    )
  )
  cd_blk.append(Store(o_buf, iadd(out_base, cd_lp), ops.prim("mul", (o_sum, inv_lt), elt)))
  c_body.append(ForRange(cd_lp, ZERO, d_dim, ONE, cd_blk, Scope("loop"), simd=True))
  if m_buf is not None:
    c_body.append(Store(m_buf, out_row, m_tot))
  if l_buf is not None:
    c_body.append(Store(l_buf, out_row, l_tot))
  c_body.append(ReleaseScope(c_scope))
  then_blk.append(IndexMap(c_shape, c_idx, c_body, c_scope, False, True, sc.stmt.loc))
  then_blk.append(ir.Release(m_part))
  then_blk.append(ir.Release(l_part))
  then_blk.append(ir.Release(o_part))
  return [If(cond, then_blk, [single_tile])]


def _match_cce_out_map(
  s: Stmt,
  z_buf: Var,
  m_buf: Var | None,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  elt: ScalarType,
  facts: IndexFacts,
) -> tuple[Var, Var | None, list[Stmt], Expr | None, str] | None:
  """Match the final Cut Cross-Entropy / LogSumExp consumer ``IndexMap(*batch, M)``:

  * ``sum_exp = reduce(add, exp(z_buf[b.., i, j]))`` (with optional ``s_buf`` store)
  * ``out[b.., i] = z_buf[b.., i, idx_j] - log(sum_exp)`` (mode ``"picked_minus_lse"``)
    OR ``out[b.., i] = log(sum_exp) + m_buf[b.., i]`` (mode ``"lse"``).

  Returns ``(out_buf, s_buf, idx_prefix_stmts, idx_j_expr, mode)``."""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) != len(batch_shape) + 1 or s.scope.owned:
    return None
  if not facts.same_shape(s.shape, (*batch_shape, m_dim)):
    return None
  body = _unwrap_nonempty_if(s.body, n_dim, facts)
  reds = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  if len(reds) != 1:
    return None
  r_pos = reds[0]
  red = body[r_pos]
  assert isinstance(red, IndexReduce)
  if len(red.shape) != 1 or not facts.same_expr(red.shape[0], n_dim) or len(red.folds) != 1:
    return None
  f = red.folds[0]
  if (
    f.op != "add"
    or f.combine
    or f.carries
    or f.a.type is not elt
    or not (isinstance(f.init, Const) and float(f.init.value) == 0.0)
  ):
    return None

  f_facts = facts.fork()
  vals: dict[int, Expr] = {}
  for t in body[:r_pos] + red.body:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      vals[t.var.uid] = t.value
  fv = _resolve_local(f.value, vals)
  expected_r = row_major_index((*s.index, red.index[0]), (*batch_shape, m_dim, n_dim))
  if not (
    isinstance(fv, Prim)
    and fv.op == "exp"
    and len(fv.args) == 1
    and isinstance(fv.args[0], Load)
    and fv.args[0].buf is z_buf
    and f_facts.same_expr(fv.args[0].index, expected_r)
  ):
    return None

  after = body[r_pos + 1 :]
  stores = [t for t in after if isinstance(t, Store)]
  if not (1 <= len(stores) <= 2) or any(not isinstance(t, (Assign, Store, Check, ir.Comment)) for t in after):
    return None
  row_flat = row_major_index(s.index, (*batch_shape, m_dim))
  s_buf: Var | None = None
  out_st: Store | None = None
  for st in stores:
    if st.atomic is not None or not f_facts.same_expr(st.index, row_flat):
      return None
    if st.value is f.result:
      s_buf = st.buf
    else:
      if out_st is not None:
        return None
      out_st = st
  if out_st is None:
    return None

  a_vals = dict(vals)
  for t in after:
    f_facts.record_stmt(t)
    if isinstance(t, Assign):
      a_vals[t.var.uid] = t.value
  ov = _resolve_local(out_st.value, a_vals)

  # Mode 1: z_buf[row_i * N + idx_j] - log(f.result)
  if isinstance(ov, Prim) and ov.op == "sub" and len(ov.args) == 2:
    lhs, rhs = ov.args
    if (
      isinstance(rhs, Prim)
      and rhs.op == "log"
      and len(rhs.args) == 1
      and rhs.args[0] is f.result
      and isinstance(lhs, Load)
      and lhs.buf is z_buf
    ):
      # Verify lhs.index == row_flat * N + idx_j (with optional bmul*i == i when dim == M!)
      # Decompose lhs.index relative to (*batch, M, N)
      if len(batch_shape) == 0:
        # 2D: i_row * N + idx_j
        if isinstance(lhs.index, Prim) and lhs.index.op == "add" and len(lhs.index.args) == 2:
          p0, p1 = lhs.index.args
          row_mul = p0
          idx_j = p1
          # Check row_mul is (i * N) or (i_89 * N) where i_89 = i * bmul
          if isinstance(row_mul, Prim) and row_mul.op == "mul" and len(row_mul.args) == 2:
            if f_facts.same_expr(row_mul.args[1], n_dim):
              i_row = row_mul.args[0]
              # Allow i_row == s.index[0] or i_row == s.index[0] * bmul (from np.arange(M) broadcast)
              i_used = _expr_uses_uids(i_row, {s.index[0].uid})
              if i_used and not _expr_uses_uids(idx_j, {red.index[0].uid}):
                prefix = [t for t in body[:r_pos] if isinstance(t, (Assign, Check))]
                return out_st.buf, s_buf, prefix, idx_j, "picked_minus_lse"

  # Mode 2: log(f.result) + m_buf[row_flat] (or m_buf + log(f.result))
  if m_buf is not None and isinstance(ov, Prim) and ov.op == "add" and len(ov.args) == 2:
    a, b = ov.args

    def is_log(ex):
      return isinstance(ex, Prim) and ex.op == "log" and len(ex.args) == 1 and ex.args[0] is f.result

    def is_m(ex):
      return isinstance(ex, Load) and ex.buf is m_buf and f_facts.same_expr(ex.index, row_flat)

    if (is_log(a) and is_m(b)) or (is_log(b) and is_m(a)):
      return out_st.buf, s_buf, [], None, "lse"

  return None


def _build_monoid_lse_tile(
  sc: _ScoreContraction,
  out_buf: Var,
  m_buf: Var | None,
  s_buf: Var | None,
  out_index: tuple[Var, ...],
  idx_prefix: list[Stmt],
  idx_j: Expr | None,
  mode: str,
  facts: IndexFacts,
) -> IndexTile:
  """Build ONE :class:`IndexTile` with 1-pass online ``(m, l)`` (+ optional
  target dot product ``picked``) CPU fallback AND :class:`MonoidalTilePlan`
  (``kind="monoid_lse"``) for GPU."""
  elt = sc.elt
  shape = (*sc.batch_shape, sc.m_dim)
  tile_shape: tuple[Expr | None, ...] = (ONE,) * len(shape)
  origin = tuple(Var("lo", I) for _ in shape)
  extent = tuple(Var("le", I) for _ in shape)
  b_vars = origin[:-1]
  i_var = origin[-1]

  h_st = _rebind_stage_batch(sc.q_stage, sc.batch_index, b_vars)
  w_st = _rebind_stage_batch(sc.k_stage, sc.batch_index, b_vars)
  bias_st = None if sc.bias_stage is None else _rebind_stage_batch(sc.bias_stage, sc.batch_index, b_vars)
  xq_st = None if sc.ipa_xq_stage is None else _rebind_stage_batch(sc.ipa_xq_stage, sc.batch_index, b_vars)
  xk_st = None if sc.ipa_xk_stage is None else _rebind_stage_batch(sc.ipa_xk_stage, sc.batch_index, b_vars)

  i_m, j_m = Var("lmi", I), Var("lmj", I)
  m_subst = {a.uid: b for a, b in zip(sc.batch_index, b_vars)}
  m_subst[sc.i_var.uid] = i_m
  m_subst[sc.j_var.uid] = j_m
  m_cl = Cloner(m_subst)
  mask_cond = None if sc.mask_cond is None else m_cl.expr(sc.mask_cond)
  mask_index = None if sc.mask_index is None else m_cl.expr(sc.mask_index)
  tile_false_g, tile_true_g, tile_ov = _derive_law5_guards(mask_cond, i_m, j_m, facts)

  body: list[Stmt] = []
  scope = Scope("loop")
  row_flat = row_major_index((*b_vars, i_var), shape)
  zero_c = Const(0.0, elt)
  one_c = Const(1.0, elt)

  # Clone target index computation (and bounds checks) onto origin
  tgt_var: Var | None = None
  aux_exprs: dict[str, Expr] = {}
  if idx_j is not None:
    p_cl = Cloner({v.uid: o for v, o in zip(out_index, origin)})
    for v in defined(idx_prefix).values():
      p_cl.var(v)
    for st in idx_prefix:
      body.append(p_cl.stmt(st))
    tgt_var = Var("ltgt", I)
    body.append(Assign(tgt_var, p_cl.expr(idx_j)))
    # Also try to extract simple target load for GPU kernel (Load(tbuf, offset + i))
    t_vals = {t.var.uid: t.value for t in idx_prefix if isinstance(t, Assign)}
    res_j = _resolve_local(idx_j, t_vals)
    t_load: Load | None = None
    for x in ir.walk_expr(res_j):
      if isinstance(x, Load):
        t_load = x
        break
    if t_load is not None:
      aux_exprs["target_buf"] = t_load.buf
      t_cl = Cloner({v.uid: o for v, o in zip(out_index, origin)})
      aux_exprs["target_index"] = t_cl.expr(t_load.index)

  m_acc = Var("lm_acc", elt)
  l_acc = Var("ll_acc", elt)
  picked_acc = Var("lpick_acc", elt)
  body.append(Assign(m_acc, Const(float("-inf"), elt)))
  body.append(Assign(l_acc, zero_c))
  if tgt_var is not None:
    body.append(Assign(picked_acc, zero_c))

  j_lp = Var("lj", I)
  j_blk: list[Stmt] = []
  k_lp = Var("lk", I)
  s_acc = Var("ls_acc", elt)
  j_blk.append(Assign(s_acc, zero_c))
  k_blk: list[Stmt] = []
  hv = _inline_stage(h_st, (i_var, k_lp), k_blk)
  wv = _inline_stage(w_st, (j_lp, k_lp), k_blk)
  if sc.score_op == "sqdist":
    diff = Var("ldiff", elt)
    k_blk.append(Assign(diff, ops.prim("sub", (hv, wv), elt)))
    term: Expr = ops.prim("mul", (diff, diff), elt)
  else:
    term = ops.prim("mul", (hv, wv), elt)
  k_blk.append(Assign(s_acc, ops.prim("add", (s_acc, term), elt)))
  j_blk.append(ForRange(k_lp, ZERO, sc.k_dim, ONE, k_blk, Scope("loop"), simd=True, reduction=("add", [s_acc])))

  s_val: Expr = s_acc
  if sc.score_scale is not None:
    s_val = ops.prim("mul", (s_val, cast(sc.score_scale, elt)), elt)
  if xq_st is not None and xk_st is not None and sc.ipa_c_dim is not None and sc.ipa_gamma is not None:
    c_lp = Var("fc", I)
    sq_acc = Var("fsq_acc", elt)
    j_blk.append(Assign(sq_acc, zero_c))
    c_blk: list[Stmt] = []
    xqv = _inline_stage(xq_st, (i_var, c_lp), c_blk)
    xkv = _inline_stage(xk_st, (j_lp, c_lp), c_blk)
    cdf = Var("fcdf", elt)
    c_blk.append(Assign(cdf, ops.prim("sub", (xqv, xkv), elt)))
    c_blk.append(Assign(sq_acc, ops.prim("add", (sq_acc, ops.prim("mul", (cdf, cdf), elt)), elt)))
    j_blk.append(ForRange(c_lp, ZERO, sc.ipa_c_dim, ONE, c_blk, Scope("loop"), simd=True, reduction=("add", [sq_acc])))
    s_val = ops.prim("sub", (s_val, ops.prim("mul", (cast(sc.ipa_gamma, elt), sq_acc), elt)), elt)
  if bias_st is not None:
    bv = _inline_stage(bias_st, (i_var, j_lp), j_blk)
    s_val = ops.prim("add", (s_val, cast(bv, elt)), elt)
  if sc.mask_fill is not None and (mask_cond is not None or sc.mask_buf is not None):
    c_cl = Cloner({i_m.uid: i_var, j_m.uid: j_lp})
    if sc.mask_buf is not None and mask_index is not None:
      cond_at: Expr = Load(sc.mask_buf, c_cl.expr(mask_index))
    else:
      assert mask_cond is not None
      cond_at = c_cl.expr(mask_cond)
    s_val = ops.select(cond_at, s_val, cast(sc.mask_fill, elt))
  s_cur = Var("ls_cur", elt)
  j_blk.append(Assign(s_cur, s_val))

  m_new = Var("lm_new", elt)
  is_first = cmp("eq", j_lp, ZERO)
  j_blk.append(Assign(m_new, ops.select(is_first, s_cur, ops.prim("max", (m_acc, s_cur), elt))))
  j_blk.append(
    Assign(
      l_acc,
      ops.select(
        is_first,
        one_c,
        ops.prim(
          "add",
          (
            ops.prim("mul", (l_acc, ops.prim("exp", (ops.prim("sub", (m_acc, m_new), elt),), elt)), elt),
            ops.prim("exp", (ops.prim("sub", (s_cur, m_new), elt),), elt),
          ),
          elt,
        ),
      ),
    )
  )
  j_blk.append(Assign(m_acc, m_new))
  if tgt_var is not None:
    j_blk.append(Assign(picked_acc, ops.select(cmp("eq", j_lp, tgt_var), s_cur, picked_acc)))
  body.append(ForRange(j_lp, ZERO, sc.n_dim, ONE, j_blk, Scope("loop")))

  lse_expr = ops.prim("add", (m_acc, ops.prim("log", (l_acc,), elt)), elt)
  out_bufs: dict[str, Var] = {}
  if m_buf is not None:
    body.append(Store(m_buf, row_flat, m_acc))
    out_bufs["m"] = m_buf
  if s_buf is not None:
    body.append(Store(s_buf, row_flat, l_acc))
    out_bufs["l"] = s_buf
  if mode == "picked_minus_lse" and tgt_var is not None:
    # (picked - m) - log(l) == picked - (m + log(l))
    body.append(Store(out_buf, row_flat, ops.prim("sub", (picked_acc, lse_expr), elt)))
    out_bufs["out"] = out_buf
  elif mode == "picked_minus_m" and tgt_var is not None:
    # Training form where picked = z[i, y_i] = logit[i, y_i] - m_i is stored alongside m_i and l_i!
    body.append(Store(out_buf, row_flat, ops.prim("sub", (picked_acc, m_acc), elt)))
    out_bufs["picked_sub_m"] = out_buf
  else:
    body.append(Store(out_buf, row_flat, lse_expr))
    out_bufs["lse"] = out_buf
  body.append(ReleaseScope(scope))

  # Only attach GPU monoidal_tile if no target checks or target_buf was extracted cleanly
  gpu_plan: MonoidalTilePlan | None = None
  if mode == "lse" or "target_buf" in aux_exprs:
    lse_stages: dict[str, TileStage] = {"h": h_st, "w": w_st}
    if bias_st is not None:
      lse_stages["bias"] = bias_st
    if xq_st is not None and xk_st is not None and sc.ipa_c_dim is not None and sc.ipa_gamma is not None:
      lse_stages["xq"] = xq_st
      lse_stages["xk"] = xk_st
      aux_exprs["ipa_gamma"] = cast(sc.ipa_gamma, elt)
      aux_exprs["ipa_c_dim"] = sc.ipa_c_dim
    gpu_plan = MonoidalTilePlan(
      kind="monoid_lse",
      elt=elt,
      batch_shape=sc.batch_shape,
      batch_index=b_vars,
      m_dim=sc.m_dim,
      n_dim=sc.n_dim,
      k_dim=sc.k_dim,
      stages=lse_stages,
      score_op=sc.score_op,
      score_scale=sc.score_scale,
      mask_cond=mask_cond,
      mask_index_vars=(i_m, j_m),
      mask_buf=sc.mask_buf,
      mask_index=mask_index,
      mask_fill=sc.mask_fill,
      tile_false_guard=tile_false_g,
      tile_true_guard=tile_true_g,
      tile_origin_vars=tile_ov,
      out_bufs=out_bufs,
      aux_exprs=aux_exprs,
    )
  return IndexTile(
    shape=shape,
    tile_shape=tile_shape,
    origin=origin,
    extent=extent,
    stages=[],
    body=body,
    scope=scope,
    level="shared",
    parallel=sc.stmt.parallel,
    loc=sc.stmt.loc,
    monoidal_tile=gpu_plan,
  )


def _buf_read_counts(stmts: list[Stmt], uids: set[int]) -> dict[int, int]:
  counts = {u: 0 for u in uids}
  for s in ir.walk_stmts(stmts):
    exprs = (s.index, s.value) if isinstance(s, Store) else ir.stmt_exprs(s)
    for e in exprs:
      for x in ir.walk_expr(e):
        if isinstance(x, Var) and x.uid in counts:
          counts[x.uid] += 1
  return counts


def _fuse_block(
  block: list[Stmt], facts: IndexFacts, binfo: BufferInfo, gone: set[int], owner: Scope, target=None
) -> tuple[list[Stmt], bool]:
  """Walk ``block`` looking for 2-level FlashAttention / pairwise-monoid and
  Cut Cross-Entropy / pairwise-LogSumExp chains."""
  changed = False
  for s in block:
    sub_facts = facts.fork()
    if isinstance(s, (IndexMap, IndexReduce)):
      sub_facts.activate_nonempty(s.shape)
    sub_owner = s.scope if hasattr(s, "scope") and isinstance(s.scope, Scope) else owner
    for b in ir.sub_blocks(s):
      b[:], c = _fuse_block(b, sub_facts, binfo, gone, sub_owner, target)
      changed |= c

  # Collect top-level IndexMap positions and scan for chains
  out: list[Stmt] = []
  walk_facts = facts.fork()
  k = 0
  n = len(block)
  while k < n:
    s = block[k]
    walk_facts.record_stmt(s)
    sc = _match_score_contraction(s, k, walk_facts)
    if sc is not None and sc.out_buf.uid in binfo.local_allocs and sc.out_buf.uid not in binfo.escapes:
      # Scan forward through harmless glue (Alloc, Assign, Check, Comment, Release) to find consumers
      # of sc.out_buf
      step_facts = walk_facts.fork()
      hoist_before: list[Stmt] = []
      j = k + 1

      def _scan_to_next_map(
        pos: int, f: IndexFacts, hoist: list[Stmt], skip_independent_of: set[int] | None = None
      ) -> int | None:
        while pos < n:
          t = block[pos]
          if isinstance(t, (ir.Comment, ir.Alloc, Check)):
            f.record_stmt(t)
            hoist.append(t)
            pos += 1
            continue
          if isinstance(t, Assign) and not isinstance(t.var.type, BufType) and is_pure_expr(t.value):
            f.record_stmt(t)
            hoist.append(t)
            pos += 1
            continue
          if isinstance(t, IndexMap):
            if skip_independent_of is not None:
              # Allow hoisting an independent transpose/copy map (e.g. bt = V.T.copy() before P @ V)
              if all(v == 0 for v in _buf_read_counts([t], skip_independent_of).values()):
                hoist.append(t)
                pos += 1
                continue
            return pos
          return None
        return None

      # --- Try Pattern 1A: FlashAttention (S -> [M ->] E=exp(S-m) -> [L ->] P=E/l -> O=P@V) ---
      p1 = _scan_to_next_map(j, step_facts, hoist_before)
      fused_done = False
      if p1 is not None:
        m_buf: Var | None = None
        e_buf: Var | None = None
        e_pos: int | None = None
        hoist_m: list[Stmt] = []
        f_after_e = step_facts.fork()
        matched_stmts: list[Stmt] = [block[k]]

        m_exp = _match_row_max_transform_map(
          block[p1], sc.out_buf, sc.batch_shape, sc.m_dim, sc.n_dim, step_facts, want_exp=True
        )
        if m_exp is not None:
          e_buf, m_buf, _ = m_exp
          e_pos = p1
          matched_stmts.append(block[p1])
        else:
          # Unfused form (when m_buf is also read by AD backward!): M_map followed by E_map
          m_only = _match_row_reduce_map(
            block[p1], sc.out_buf, sc.batch_shape, sc.m_dim, sc.n_dim, step_facts, want_op="max"
          )
          if m_only is not None:
            m_buf = m_only
            matched_stmts.append(block[p1])
            p1b = _scan_to_next_map(p1 + 1, f_after_e, hoist_m)
            if p1b is not None:
              eb = _match_ew_row_transform_map(
                block[p1b], sc.out_buf, m_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f_after_e, "exp_sub"
              )
              if eb is not None:
                e_buf = eb
                e_pos = p1b
                matched_stmts.append(block[p1b])

        if (
          e_buf is not None and e_pos is not None and e_buf.uid in binfo.local_allocs and e_buf.uid not in binfo.escapes
        ):
          hoist2: list[Stmt] = []
          f2 = f_after_e.fork()
          p2 = _scan_to_next_map(e_pos + 1, f2, hoist2)
          l_buf: Var | None = None
          p_buf: Var | None = None
          p_pos: int | None = None
          hoist_l: list[Stmt] = []
          f_after_p = f2.fork()
          if p2 is not None:
            m_div = _match_row_sum_div_map(block[p2], e_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f2)
            if m_div is not None:
              p_buf, l_buf = m_div
              p_pos = p2
              matched_stmts.append(block[p2])
            else:
              # Unfused form (when l_buf is also read by AD backward!): L_map followed by P_map
              l_only = _match_row_reduce_map(block[p2], e_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f2, want_op="add")
              if l_only is not None:
                l_buf = l_only
                matched_stmts.append(block[p2])
                p2b = _scan_to_next_map(p2 + 1, f_after_p, hoist_l)
                if p2b is not None:
                  pb = _match_ew_row_transform_map(
                    block[p2b], e_buf, l_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f_after_p, "div"
                  )
                  if pb is not None:
                    p_buf = pb
                    p_pos = p2b
                    matched_stmts.append(block[p2b])

          if (
            p_buf is not None
            and p_pos is not None
            and p_buf.uid in binfo.local_allocs
            and p_buf.uid not in binfo.escapes
          ):
            hoist3: list[Stmt] = []
            f3 = f_after_p.fork()
            p3 = _scan_to_next_map(p_pos + 1, f3, hoist3, skip_independent_of={sc.out_buf.uid, e_buf.uid, p_buf.uid})
            if p3 is not None:
              m_val = _match_value_contraction(block[p3], p_buf, sc.batch_shape, sc.m_dim, sc.n_dim, sc.elt, f3)
              if m_val is not None:
                o_buf, d_dim, v_stage, v_batch = m_val
                matched_stmts.append(block[p3])
                dead_uids = {sc.out_buf.uid, e_buf.uid, p_buf.uid}
                in_chain = _buf_read_counts(matched_stmts, dead_uids)
                if all(binfo.readers.get(u, 0) == in_chain[u] for u in dead_uids):
                  for h_st in hoist_before + hoist_m + hoist2 + hoist_l + hoist3:
                    if isinstance(h_st, ir.Alloc) and h_st.var.uid in dead_uids:
                      continue
                    out.append(h_st)
                  for chk in sc.pre_checks:
                    out.append(chk)
                  sc_f, v_st_f, v_b_f, gqa_pre = _maybe_fold_gqa_group(sc, v_stage, v_batch)
                  out.extend(gqa_pre)
                  tile = _build_flash_monoid_tile(sc_f, o_buf, d_dim, v_st_f, v_b_f, m_buf, l_buf, None, f3)
                  out.extend(
                    _maybe_wrap_flash_decoding(sc_f, o_buf, d_dim, v_st_f, v_b_f, m_buf, l_buf, tile, f3, owner, target)
                  )
                  gone.update(dead_uids)
                  walk_facts = f3
                  k = p3 + 1
                  changed = True
                  fused_done = True

      # --- Try Pattern 1A-train: S -> M=max(score_mod(S)) -> L=sum(exp(score_mod(S)-M)) -> P=exp(..)/L -> O=P@V ---
      if not fused_done and sc.score_scale is None and sc.mask_fill is None and sc.bias_stage is None:
        tf = walk_facts.fork()
        th1: list[Stmt] = []
        tp1 = _scan_to_next_map(j, tf, th1)
        if tp1 is not None:
          ms = _match_row_max_score_mod_map(block[tp1], sc.out_buf, sc.batch_shape, sc.m_dim, sc.n_dim, tf)
          if ms is not None:
            m_buf_t, t_scale, t_mc, t_mb, t_mi, t_mf, t_bias_e, t_ridx, t_jvar = ms
            th2: list[Stmt] = []
            tf2 = tf.fork()
            tp2 = _scan_to_next_map(tp1 + 1, tf2, th2)
            if tp2 is not None:
              l_buf_t = _match_row_sum_exp_score_mod_map(
                block[tp2], sc.out_buf, m_buf_t, sc.batch_shape, sc.m_dim, sc.n_dim, tf2
              )
              if l_buf_t is not None:
                th3: list[Stmt] = []
                tf3 = tf2.fork()
                tp3 = _scan_to_next_map(tp2 + 1, tf3, th3)
                if tp3 is not None:
                  p_buf_t = _match_ew_softmax_prob_map(
                    block[tp3], sc.out_buf, m_buf_t, l_buf_t, sc.batch_shape, sc.m_dim, sc.n_dim, tf3
                  )
                  if p_buf_t is not None and p_buf_t.uid in binfo.local_allocs and p_buf_t.uid not in binfo.escapes:
                    th4: list[Stmt] = []
                    tf4 = tf3.fork()
                    tp4 = _scan_to_next_map(tp3 + 1, tf4, th4, skip_independent_of={sc.out_buf.uid, p_buf_t.uid})
                    if tp4 is not None:
                      m_val = _match_value_contraction(
                        block[tp4], p_buf_t, sc.batch_shape, sc.m_dim, sc.n_dim, sc.elt, tf4
                      )
                      if m_val is not None:
                        o_buf, d_dim, v_stage, v_batch = m_val
                        dead_uids = {sc.out_buf.uid, p_buf_t.uid}
                        t_stmts = [block[k], block[tp1], block[tp2], block[tp3], block[tp4]]
                        in_chain = _buf_read_counts(t_stmts, dead_uids)
                        if all(binfo.readers.get(u, 0) == in_chain[u] for u in dead_uids):
                          # Rebind t_mc / t_mi / t_bias_e from (t_ridx, t_jvar) onto (sc.batch_index, sc.i_var, sc.j_var)
                          r_sub = {a.uid: b for a, b in zip(t_ridx[:-1], sc.batch_index)}
                          r_sub[t_ridx[-1].uid] = sc.i_var
                          r_sub[t_jvar.uid] = sc.j_var
                          r_cl = Cloner(r_sub)
                          t_bias_st: TileStage | None = None
                          if t_bias_e is not None:
                            t_bias_st = _build_stage(
                              r_cl.expr(t_bias_e), (sc.i_var, sc.j_var), (sc.m_dim, sc.n_dim), {}, {}, "bias"
                            )
                          sc_mod = _ScoreContraction(
                            stmt_idx=sc.stmt_idx,
                            stmt=sc.stmt,
                            out_buf=sc.out_buf,
                            elt=sc.elt,
                            batch_shape=sc.batch_shape,
                            batch_index=sc.batch_index,
                            m_dim=sc.m_dim,
                            n_dim=sc.n_dim,
                            k_dim=sc.k_dim,
                            i_var=sc.i_var,
                            j_var=sc.j_var,
                            q_stage=sc.q_stage,
                            k_stage=sc.k_stage,
                            score_op=sc.score_op,
                            score_scale=t_scale,
                            mask_cond=None if t_mc is None else r_cl.expr(t_mc),
                            mask_buf=t_mb,
                            mask_index=None if t_mi is None else r_cl.expr(t_mi),
                            mask_fill=t_mf,
                            pre_checks=sc.pre_checks,
                            bias_stage=t_bias_st,
                          )
                          for h_st in th1 + th2 + th3 + th4:
                            if isinstance(h_st, ir.Alloc) and h_st.var.uid in dead_uids:
                              continue
                            out.append(h_st)
                          for chk in sc.pre_checks:
                            out.append(chk)
                          tile = _build_flash_monoid_tile(
                            sc_mod, o_buf, d_dim, v_stage, v_batch, m_buf_t, l_buf_t, None, tf4
                          )
                          out.append(tile)
                          gone.update(dead_uids)
                          walk_facts = tf4
                          k = tp4 + 1
                          changed = True
                          fused_done = True

      if fused_done:
        continue

      # --- Try Pattern 1B: Cut Cross-Entropy / LogSumExp (S -> [M ->] Z=S-m -> [S_exp ->] out=picked-log(s)) ---
      step_facts = walk_facts.fork()
      hoist_before = []
      p1 = _scan_to_next_map(j, step_facts, hoist_before)
      if p1 is not None:
        m_buf = None
        z_buf: Var | None = None
        z_pos: int | None = None
        hoist_m = []
        f_after_z = step_facts.fork()
        c_stmts: list[Stmt] = [block[k]]

        m_sub = _match_row_max_transform_map(
          block[p1], sc.out_buf, sc.batch_shape, sc.m_dim, sc.n_dim, step_facts, want_exp=False
        )
        if m_sub is not None:
          z_buf, m_buf, _ = m_sub
          z_pos = p1
          c_stmts.append(block[p1])
        else:
          m_only = _match_row_reduce_map(
            block[p1], sc.out_buf, sc.batch_shape, sc.m_dim, sc.n_dim, step_facts, want_op="max"
          )
          if m_only is not None:
            m_buf = m_only
            c_stmts.append(block[p1])
            p1b = _scan_to_next_map(p1 + 1, f_after_z, hoist_m)
            if p1b is not None:
              zb = _match_ew_row_transform_map(
                block[p1b], sc.out_buf, m_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f_after_z, "sub"
              )
              if zb is not None:
                z_buf = zb
                z_pos = p1b
                c_stmts.append(block[p1b])
              else:
                # Training order: S -> M -> S_exp=sum(exp(S-M)) -> Z=S-M -> Picked=Z[i, tgt[i]]
                s_exp_buf = _match_row_sum_exp_score_mod_map(
                  block[p1b], sc.out_buf, m_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f_after_z
                )
                if s_exp_buf is not None and len(sc.batch_shape) == 0:
                  hoist_z: list[Stmt] = []
                  f_z = f_after_z.fork()
                  p1c = _scan_to_next_map(p1b + 1, f_z, hoist_z)
                  if p1c is not None:
                    zb2 = _match_ew_row_transform_map(
                      block[p1c], sc.out_buf, m_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f_z, "sub"
                    )
                    if zb2 is not None and zb2.uid in binfo.local_allocs and zb2.uid not in binfo.escapes:
                      hoist_pk: list[Stmt] = []
                      f_pk = f_z.fork()
                      p1d = _scan_to_next_map(p1c + 1, f_pk, hoist_pk)
                      if p1d is not None:
                        # Check if a shape-equality check ``dim == m_dim`` appears right after p1d (from picked - log_s)
                        post_chks: list[Stmt] = []
                        f_post = f_pk.fork()
                        p_after = p1d + 1
                        while p_after < n and isinstance(block[p_after], (Check, ir.Comment)):
                          f_post.record_stmt(block[p_after])
                          post_chks.append(block[p_after])
                          p_after += 1
                        pk_map_cand = block[p1d]
                        assert isinstance(pk_map_cand, IndexMap)
                        shape_ok = f_post.same_shape(pk_map_cand.shape, (sc.m_dim,))
                        pk_m = (
                          _match_cce_picked_map(
                            pk_map_cand, zb2, sc.m_dim, sc.n_dim, f_post, allow_dim_eq_check=shape_ok
                          )
                          if shape_ok
                          else None
                        )
                        if pk_m is not None:
                          pk_buf, idx_prefix, idx_j = pk_m
                          dead_uids = {sc.out_buf.uid, zb2.uid}
                          t_stmts = [block[k], block[p1], block[p1b], block[p1c], block[p1d]]
                          in_chain = _buf_read_counts(t_stmts, dead_uids)
                          if all(binfo.readers.get(u, 0) == in_chain[u] for u in dead_uids):
                            for h_st in hoist_before + hoist_m + hoist_z + hoist_pk + post_chks:
                              if isinstance(h_st, ir.Alloc) and h_st.var.uid in dead_uids:
                                continue
                              out.append(h_st)
                            for chk in sc.pre_checks:
                              out.append(chk)
                            tile = _build_monoid_lse_tile(
                              sc,
                              pk_buf,
                              m_buf,
                              s_exp_buf,
                              pk_map_cand.index,
                              idx_prefix,
                              idx_j,
                              "picked_minus_m",
                              f_post,
                            )
                            out.append(tile)
                            gone.update(dead_uids)
                            walk_facts = f_post
                            k = p_after
                            changed = True
                            continue

        if (
          z_buf is not None and z_pos is not None and z_buf.uid in binfo.local_allocs and z_buf.uid not in binfo.escapes
        ):
          hoist2 = []
          f2 = f_after_z.fork()
          p2 = _scan_to_next_map(z_pos + 1, f2, hoist2)
          if p2 is not None:
            m_out = _match_cce_out_map(block[p2], z_buf, m_buf, sc.batch_shape, sc.m_dim, sc.n_dim, sc.elt, f2)
            if m_out is not None:
              out_buf, s_buf, idx_prefix, idx_j, mode = m_out
              c_stmts.append(block[p2])
              dead_uids = {sc.out_buf.uid, z_buf.uid}
              in_chain = _buf_read_counts(c_stmts, dead_uids)
              if all(binfo.readers.get(u, 0) == in_chain[u] for u in dead_uids):
                for h_st in hoist_before + hoist_m + hoist2:
                  if isinstance(h_st, ir.Alloc) and h_st.var.uid in dead_uids:
                    continue
                  out.append(h_st)
                for chk in sc.pre_checks:
                  out.append(chk)
                out_map = block[p2]
                assert isinstance(out_map, IndexMap)
                tile = _build_monoid_lse_tile(sc, out_buf, m_buf, s_buf, out_map.index, idx_prefix, idx_j, mode, f2)
                out.append(tile)
                gone.update(dead_uids)
                walk_facts = f2
                k = p2 + 1
                changed = True
                continue
            else:
              # Unfused form (when s_buf = sum(exp(z)) is also read by AD backward!): S_map followed by Out_map
              s_only = _match_row_reduce_map(
                block[p2], z_buf, sc.batch_shape, sc.m_dim, sc.n_dim, f2, want_op="add", want_exp_input=True
              )
              if s_only is not None:
                c_stmts.append(block[p2])
                hoist_s: list[Stmt] = []
                f_after_s = f2.fork()
                p2b = _scan_to_next_map(p2 + 1, f_after_s, hoist_s)
                if p2b is not None:
                  g_out = _match_cce_gather_only_out_map(
                    block[p2b], z_buf, s_only, sc.batch_shape, sc.m_dim, sc.n_dim, f_after_s
                  )
                  if g_out is not None:
                    out_buf, idx_prefix, idx_j = g_out
                    c_stmts.append(block[p2b])
                    dead_uids = {sc.out_buf.uid, z_buf.uid}
                    in_chain = _buf_read_counts(c_stmts, dead_uids)
                    if all(binfo.readers.get(u, 0) == in_chain[u] for u in dead_uids):
                      for h_st in hoist_before + hoist_m + hoist2 + hoist_s:
                        if isinstance(h_st, ir.Alloc) and h_st.var.uid in dead_uids:
                          continue
                        out.append(h_st)
                      for chk in sc.pre_checks:
                        out.append(chk)
                      out_map = block[p2b]
                      assert isinstance(out_map, IndexMap)
                      tile = _build_monoid_lse_tile(
                        sc, out_buf, m_buf, s_only, out_map.index, idx_prefix, idx_j, "picked_minus_lse", f_after_s
                      )
                      out.append(tile)
                      gone.update(dead_uids)
                      walk_facts = f_after_s
                      k = p2b + 1
                      changed = True
                      continue

    out.append(s)
    k += 1
  return out, changed


def fuse_monoidal_tiles(fn: ir.Function, options) -> None:
  """Fuse contraction + rescaled-monoid chains into 1-pass SRAM-tiled
  :class:`~parakeet.syntax.ir.IndexTile` operators (Laws 1 & 5)."""
  if not options.fusion or not options.tiling:
    return
  target = getattr(options, "target", None)
  for _ in range(4):
    gone: set[int] = set()
    binfo = BufferInfo.analyze(fn)
    facts = IndexFacts.from_function(fn)
    fn.body[:], changed = _fuse_block(fn.body, facts, binfo, gone, fn.scope, target)
    if gone:
      drop_buffers(fn, gone, drop_allocs=True)
    if not changed:
      break


__all__ = ["fuse_monoidal_tiles"]
