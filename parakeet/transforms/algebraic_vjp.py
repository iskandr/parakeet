"""Law 2: Algebraic Rematerialization & Compact VJPs on the AD Tape.

Replaces naive multi-node reverse-mode differentiation of frontier algebraic
compositions with exact O(N)-memory rematerialized VJPs:

1. **FlashAttention Rematerialized VJP** (``O = softmax(score_mod(Q @ K^T, mask)) @ V``):
   Drops all backward references to ``S`` and ``P`` (so Law 1
   :mod:`~parakeet.transforms.monoid_tile` eliminates ``O(B*H*T^2)`` HBM in
   the forward pass too!), computes ``D_i = sum_d dO_{id} O_{id}`` and
   ``lse_i = m_i + log(l_i)``, and emits ONE ``IndexTile`` carrying
   :class:`~parakeet.syntax.ir.MonoidalTilePlan` (``kind="flash_monoid_bwd"``)
   that recomputes ``S_{ij}, P_{ij}`` on-chip in SRAM with Law 5 tile-interval
   mask pruning.

2. **Cut Cross-Entropy Rematerialized VJP** (``logits = H @ W``,
   ``picked - log(sum(exp(logits - max(logits))))``):
   Drops all backward references to ``logits`` and ``z`` (so Law 1 eliminates
   ``O(BT * V)`` HBM in forward too!) and emits ONE ``IndexTile`` carrying
   ``MonoidalTilePlan(kind="cce_bwd")`` that recomputes ``p_{iv}`` in SRAM
   tiles and accumulates ``dH`` and ``dW``.

3. **Algebraic Softmax & LogSumExp VJPs**:
   For standalone ``softmax(x)`` and ``logsumexp(x)`` not fused into
   FlashAttention/CCE, eliminates the shift-invariant ``max`` backward
   (``hit/count/share``) and emits ``p * (g - sum(g * p))`` or ``g * exp(x - lse)``.

4. **1-Pass LayerNorm / RMSNorm VJP**:
   Collapses the 9-node mean/variance/rsqrt backward chain into the exact
   closed-form ``inv_std * (g - mean(g) - x_hat * mean(g * x_hat))``.
"""

from __future__ import annotations

from ..ndtypes import Bool, BufType, ScalarType
from ..syntax import ir, ops
from ..syntax.clone import Cloner, defined
from ..syntax.ir import (
  Assign,
  Const,
  Expr,
  ForRange,
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
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, imul
from ..syntax.values import ArrayVal
from .analysis import IndexFacts, TileInterval, tile_predicate_guards
from .lowering import row_major_index

# ---------------------------------------------------------------------------
# Elementwise function prober (zero heuristics — exact IR inspection)
# ---------------------------------------------------------------------------


def _probe_ew(el, node) -> tuple[str | None, Expr | None, list[Var]]:
  """Probe a tape ``elementwise`` node's scalar function with dummy variables.
  Returns ``(op_kind, result_expr, probes)`` where ``op_kind`` is one of
  ``"add"``, ``"sub"``, ``"mul"``, ``"div"``, ``"exp"``, ``"log"``, ``"sqrt"``,
  ``"pow2"``, ``"select"``, or ``None``."""
  if node.kind != "elementwise":
    return None, None, []
  _active, fn, _elt, _cost = node.info
  probes = [Var("p", x.elt if isinstance(x, ArrayVal) else x.type) for x in node.inputs]
  try:
    with el.trial():
      res = fn(*probes)
  except Exception:  # noqa: BLE001
    return None, None, []
  if not isinstance(res, Prim):
    return None, res, probes
  op = res.op
  if op in ("add", "sub", "mul", "div", "exp", "log", "sqrt", "select"):
    return op, res, probes
  if op == "pow" and len(res.args) == 2:
    if (isinstance(res.args[1], Const) and float(res.args[1].value) == 2.0) or (
      len(node.inputs) == 2 and isinstance(node.inputs[1], Const) and float(node.inputs[1].value) == 2.0
    ):
      return "pow2", res, probes
  return None, res, probes


def _unwrap_alias(node_idx: int, nodes: list, prod_by_id: dict[int, int]) -> int:
  """Follow backward through ``alias`` nodes to the producing operator node."""
  cur = node_idx
  seen: set[int] = set()
  while cur not in seen and nodes[cur].kind == "alias":
    seen.add(cur)
    inp = nodes[cur].inputs[0]
    nxt = prod_by_id.get(id(inp))
    if nxt is None:
      break
    cur = nxt
  return cur


def _same_val(a, b) -> bool:
  if a is b:
    return True
  if isinstance(a, ArrayVal) and isinstance(b, ArrayVal):
    if a.buf is not None and b.buf is not None and a.buf is b.buf:
      return a.shape == b.shape and ir.same(a.offset, b.offset) and a.strides == b.strides
  return False


def _axis_int(el, axis, rank: int) -> int | None:
  if isinstance(axis, Const):
    a = int(axis.value)
    return a + rank if a < 0 else a
  if isinstance(axis, tuple) and len(axis) == 1 and isinstance(axis[0], Const):
    a = int(axis[0].value)
    return a + rank if a < 0 else a
  return None


def _stage_from_array(
  el, arr: ArrayVal, batch_vars: tuple[Var, ...], prefix: str, transposed_last2: bool = False
) -> TileStage:
  r_var = Var(f"{prefix}_r", I)
  c_var = Var(f"{prefix}_c", I)
  coords = (*batch_vars, c_var, r_var) if transposed_last2 else (*batch_vars, r_var, c_var)
  shape = (arr.shape[-1], arr.shape[-2]) if transposed_last2 else (arr.shape[-2], arr.shape[-1])
  st_body: list[Stmt] = []
  st_scope = Scope("stage")
  with el.in_block(st_body, st_scope, loop=True):
    val = el.elem(arr, coords)
  return TileStage(Var(f"st_{prefix}", BufType(arr.elt)), shape, (r_var, c_var), st_body, val, st_scope)


def _inline_stage(st: TileStage, coords: tuple[Expr, ...], out_stmts: list[Stmt]) -> Expr:
  cl = Cloner({v.uid: c for v, c in zip(st.index, coords)})
  for v in defined(st.body).values():
    cl.var(v)
  out_stmts.extend(cl.stmt(t) for t in st.body)
  return cl.expr(st.value)


# ---------------------------------------------------------------------------
# Subgraph matchers
# ---------------------------------------------------------------------------


def _match_softmax_chain(el, nodes, prod_by_id, end_idx: int):
  """Match ``m = max(X, ax, kd=True) -> z = X - m -> e = exp(z) -> l = sum(e, ax, kd=True) -> p = e / l``
  (with optional trailing ``alias`` at ``end_idx``).
  Returns ``(x_val, m_node, l_node, p_out, ax, all_node_indices)`` or ``None``."""
  div_idx = _unwrap_alias(end_idx, nodes, prod_by_id)
  alias_idxs = [end_idx] if end_idx != div_idx else []
  n_div = nodes[div_idx]
  op_div, _, _ = _probe_ew(el, n_div)
  if op_div != "div" or len(n_div.inputs) != 2:
    return None
  e_val, l_val = n_div.inputs
  e_idx = prod_by_id.get(id(e_val))
  l_idx = prod_by_id.get(id(l_val))
  if e_idx is None or l_idx is None:
    return None
  n_sum = nodes[l_idx]
  if n_sum.kind != "reduce":
    return None
  r_op, r_ax, r_kd, r_act_a, r_act_i = n_sum.info
  if r_op != "add" or not r_kd or not r_act_a or r_act_i:
    return None
  if n_sum.inputs[0] is not e_val:
    return None

  n_exp = nodes[e_idx]
  op_exp, _, _ = _probe_ew(el, n_exp)
  if op_exp != "exp" or len(n_exp.inputs) != 1:
    return None
  z_val = n_exp.inputs[0]
  z_idx = prod_by_id.get(id(z_val))
  if z_idx is None:
    return None
  n_sub = nodes[z_idx]
  op_sub, _, _ = _probe_ew(el, n_sub)
  if op_sub != "sub" or len(n_sub.inputs) != 2:
    return None
  x_val, m_val = n_sub.inputs
  m_idx = prod_by_id.get(id(m_val))
  if m_idx is None:
    return None
  n_max = nodes[m_idx]
  if n_max.kind != "reduce":
    return None
  m_op, m_ax, m_kd, m_act_a, m_act_i = n_max.info
  if m_op != "max" or not m_kd or not m_act_a or m_act_i:
    return None
  if not _same_val(n_max.inputs[0], x_val):
    return None
  if not isinstance(x_val, ArrayVal):
    return None
  ax_m = _axis_int(el, m_ax, x_val.rank)
  ax_r = _axis_int(el, r_ax, x_val.rank)
  if ax_m is None or ax_m != ax_r:
    return None
  idxs = [m_idx, z_idx, e_idx, l_idx, div_idx, *alias_idxs]
  return x_val, n_max, n_sum, nodes[end_idx].out, ax_m, idxs


def _match_batched_matmul(el, nodes, prod_by_id, red_idx: int):
  """Match either 2D ``matmul(A, B)`` or N-D batched matmul ``reduce("add", (rank,), elementwise(mul, A, B))``.
  Returns ``(A_val, B_val, node_indices)`` or ``None``."""
  n = nodes[red_idx]
  if n.kind == "matmul":
    a, b = n.inputs
    if isinstance(a, ArrayVal) and isinstance(b, ArrayVal) and a.rank == 2 and b.rank == 2:
      return a, b, [red_idx]
    return None
  if n.kind == "reduce":
    r_op, r_ax, r_kd, r_act_a, r_act_i = n.info
    if r_op != "add" or r_kd or not r_act_a or r_act_i:
      return None
    mul_val = n.inputs[0]
    mul_idx = prod_by_id.get(id(mul_val))
    if mul_idx is None:
      return None
    n_mul = nodes[mul_idx]
    op_mul, _, _ = _probe_ew(el, n_mul)
    if op_mul != "mul" or len(n_mul.inputs) != 2:
      return None
    a, b = n_mul.inputs
    if not (isinstance(a, ArrayVal) and isinstance(b, ArrayVal)):
      return None
    # Batched matmul broadcasts (..., M, 1, K) * (..., 1, N, K) -> (..., M, N, K) then sums axis=-1
    if a.rank != b.rank or a.rank < 3:
      return None
    ax = _axis_int(el, r_ax, a.rank)
    if ax != a.rank - 1:
      return None
    # Recover un-expanded A (..., M, K) and B (..., K, N) from their expanded views!
    # In numpy_lib.matmul_plain, A was expanded at axis -2 (stride 0 on -2) and B was transposed + expanded at -3 (stride 0 on -3).
    if a.buf is None or b.buf is None or a.strides is None or b.strides is None:
      return None
    # Build views: A_2d = drop axis -2 (which has shape 1), B_kt = drop axis -3 (which has shape 1) -> shape (..., N, K)
    if not (isinstance(a.shape[-2], Const) and a.shape[-2].value == 1):
      return None
    if not (isinstance(b.shape[-3], Const) and b.shape[-3].value == 1):
      return None
    a_view = ArrayVal(
      a.elt,
      (*a.shape[:-2], a.shape[-1]),
      buf=a.buf,
      offset=a.offset,
      strides=(*a.strides[:-2], a.strides[-1]),
      roots=a.roots,
      contig=False,
    )
    a_view.inj = a.inj
    b_kt_view = ArrayVal(
      b.elt,
      (*b.shape[:-3], b.shape[-2], b.shape[-1]),
      buf=b.buf,
      offset=b.offset,
      strides=(*b.strides[:-3], b.strides[-2], b.strides[-1]),
      roots=b.roots,
      contig=False,
    )
    b_kt_view.inj = b.inj
    # B in matmul(A, B) has shape (..., K, N), so b_kt_view is B^T (..., N, K)
    b_view = ArrayVal(
      b.elt,
      (*b.shape[:-3], b.shape[-1], b.shape[-2]),
      buf=b.buf,
      offset=b.offset,
      strides=(*b.strides[:-3], b.strides[-1], b.strides[-2]),
      roots=b.roots,
      contig=False,
    )
    b_view.inj = b.inj
    return a_view, b_view, [mul_idx, red_idx]
  return None


# ---------------------------------------------------------------------------
# Backward handlers
# ---------------------------------------------------------------------------


def _bwd_softmax(tape, el, node, g: ArrayVal) -> None:
  from ..lib.numpy_lib import _reduce_prim as reduce_plain
  from .autodiff import UF

  (x_val,) = node.inputs
  p_out, ax = node.info[1]
  tape.stale(el, node, p_out)
  if g.buf is None:
    g = el.as_concrete(g)
  gp = el.apply_ufunc(UF["mul"], [g, p_out])
  dot = reduce_plain(el, "add", gp, Const(ax, I), Const(0, p_out.elt), "sum", True)
  dx = el.apply_ufunc(UF["mul"], [p_out, el.apply_ufunc(UF["sub"], [g, dot])])
  tape.add(el, x_val, dx)


def _bwd_logsumexp(tape, el, node, g: ArrayVal) -> None:
  from .autodiff import UF

  (x_val,) = node.inputs
  lse_out, ax, kd = node.info[1]
  tape.stale(el, node, x_val)
  tape.stale(el, node, lse_out)
  gb = g if (kd or not isinstance(g, ArrayVal)) else el.expand_dims(g, [ax])
  lb = lse_out if kd else el.expand_dims(lse_out, [ax])
  p = el.apply_ufunc(UF["exp"], [el.apply_ufunc(UF["sub"], [x_val, lb])])
  dx = el.apply_ufunc(UF["mul"], [gb, p])
  tape.add(el, x_val, dx)


def _bwd_norm(tape, el, node, g: ArrayVal) -> None:
  from ..lib.numpy_lib import _reduce_prim as reduce_plain
  from .autodiff import UF

  (x_val,) = node.inputs
  std_val, x_hat, ax, is_layernorm = node.info[1]
  tape.stale(el, node, std_val)
  tape.stale(el, node, x_hat)
  elt = x_val.elt
  if g.buf is None:
    g = el.as_concrete(g)
  if x_hat.buf is None:
    x_hat = el.as_concrete(x_hat)
  inv_n = el.tmp(ops.prim("div", (Const(1.0, elt), cast(x_val.shape[ax], elt)), elt), "inv_n")
  inv_std = el.apply_ufunc(UF["div"], [Const(1.0, elt), std_val])
  gx = el.apply_ufunc(UF["mul"], [g, x_hat])
  gx_mean = el.apply_ufunc(UF["mul"], [reduce_plain(el, "add", gx, Const(ax, I), Const(0, elt), "sum", True), inv_n])
  if is_layernorm:
    g_mean = el.apply_ufunc(UF["mul"], [reduce_plain(el, "add", g, Const(ax, I), Const(0, elt), "sum", True), inv_n])
    inner = el.apply_ufunc(UF["sub"], [el.apply_ufunc(UF["sub"], [g, g_mean]), el.apply_ufunc(UF["mul"], [x_hat, gx_mean])])
  else:
    inner = el.apply_ufunc(UF["sub"], [g, el.apply_ufunc(UF["mul"], [x_hat, gx_mean])])
  dx = el.apply_ufunc(UF["mul"], [inv_std, inner])
  tape.add(el, x_val, dx)


def _bwd_flash_attn(tape, el, node, g: ArrayVal) -> None:
  """Law 2 FlashAttention rematerialized backward (D_i form):
  reads ONLY (Q, K, V, O, m, l, dO) — ZERO N x N tensors saved!"""
  from ..lib.numpy_lib import _reduce_prim as reduce_plain
  from .autodiff import UF

  q_orig, kt_orig, v_orig = node.inputs
  (
    q_view,
    k_view,
    v_view,
    o_out,
    m_out,
    l_out,
    scale_expr,
    mask_arr,
    mask_fill,
    active_q,
    active_k,
    active_v,
  ) = node.info[1]
  tape.stale(el, node, q_view)
  tape.stale(el, node, k_view)
  tape.stale(el, node, v_view)
  tape.stale(el, node, o_out)
  tape.stale(el, node, m_out)
  tape.stale(el, node, l_out)
  if mask_arr is not None:
    tape.stale(el, node, mask_arr)

  elt: ScalarType = q_view.elt
  if g.buf is None or not g.contig:
    g = el.as_concrete(g)
  if o_out.buf is None:
    o_out = el.as_concrete(o_out)

  batch_shape = q_view.shape[:-2]
  m_dim = q_view.shape[-2]
  k_dim = q_view.shape[-1]
  n_dim = k_view.shape[-2]
  d_dim = v_view.shape[-1]

  # 1. O(B*H*T) pre-pass: lse = m + log(l), D = sum(dO * O, axis=-1)
  lse_arr = el.as_concrete(el.apply_ufunc(UF["add"], [m_out, el.apply_ufunc(UF["log"], [l_out])]))
  d_arr = el.as_concrete(
    reduce_plain(el, "add", el.apply_ufunc(UF["mul"], [g, o_out]), Const(o_out.rank - 1, I), Const(0, elt), "sum", False)
  )

  # Allocate dense zero-initialized gradients dQ, dK, dV
  dq_arr = el.alloc_array(elt, (*batch_shape, m_dim, k_dim), zero=True, label="dq")
  dk_arr = el.alloc_array(elt, (*batch_shape, n_dim, k_dim), zero=True, label="dk")
  dv_arr = el.alloc_array(elt, (*batch_shape, n_dim, d_dim), zero=True, label="dv")

  # Build IndexTile parallel over (*batch_shape,) on CPU (zero data races!),
  # with MonoidalTilePlan(kind="flash_monoid_bwd") for SRAM-tiled GPU execution!
  tile_shape_all = (*batch_shape, n_dim) if batch_shape else (n_dim,)
  # Parallel over batch dims on CPU (tile_shape=1 on batch, None on n_dim so one thread owns each (b,h) slice)
  if batch_shape:
    t_shape: tuple[Expr | None, ...] = (*(ONE,) * len(batch_shape), None)
  else:
    t_shape = (n_dim,)
  origin = tuple(Var("fbo", I) for _ in tile_shape_all)
  extent = tuple(Var("fbe", I) for _ in tile_shape_all)
  b_vars = origin[:-1] if batch_shape else ()

  q_st = _stage_from_array(el, q_view, b_vars, "bq")
  k_st = _stage_from_array(el, k_view, b_vars, "bk")
  v_st = _stage_from_array(el, v_view, b_vars, "bv")
  do_st = _stage_from_array(el, g, b_vars, "bdo")

  # Extract mask buffer/index (or symbolic predicate if mask_arr is delayed!)
  i_m, j_m = Var("bmi", I), Var("bmj", I)
  mask_buf: Var | None = None
  mask_index: Expr | None = None
  mask_cond: Expr | None = None
  if mask_arr is not None:
    if mask_arr.buf is not None:
      mask_buf = mask_arr.buf
      mask_coords = (*b_vars, i_m, j_m)[-mask_arr.rank :]
      mask_index = el.index_expr(mask_arr, mask_coords)
    else:
      mask_coords = (*b_vars, i_m, j_m)[-mask_arr.rank :]
      with el.trial():
        mc = el.elem(mask_arr, mask_coords)
      if not any(isinstance(x, Load) for x in ir.walk_expr(mc)):
        mask_cond = mc
      else:
        mc_mat = el.as_concrete(mask_arr)
        mask_buf = mc_mat.buf
        mask_index = el.index_expr(mc_mat, mask_coords)

  tile_false_g = None
  tile_true_g = None
  tile_ov = None
  if mask_cond is not None:
    i0_v, j0_v = Var("pk_ti0", I), Var("pk_tj0", I)
    iv_map = {
      i_m.uid: TileInterval(i0_v, iadd(i0_v, Const(31, I))),
      j_m.uid: TileInterval(j0_v, iadd(j0_v, Const(31, I))),
    }
    af, at = tile_predicate_guards(mask_cond, iv_map, IndexFacts())
    tile_false_g = None if (isinstance(af, Const) and not af.value) else af
    tile_true_g = None if (isinstance(at, Const) and not at.value) else at
    if tile_false_g is not None or tile_true_g is not None:
      tile_ov = (i0_v, j0_v)

  # --- CPU fallback body: outer j in [0, N), inner i in [0, M) (zero races per (b,h)!) ---
  body: list[Stmt] = []
  scope = Scope("loop")
  zero_c = Const(0.0, elt)
  scale_c = cast(scale_expr, elt) if scale_expr is not None else None

  j_lp = Var("fbj", I)
  j_blk: list[Stmt] = []
  i_lp = Var("fbi", I)
  i_blk: list[Stmt] = []

  # s_ij = dot(Q[i], K[j])
  s_acc = Var("fbs", elt)
  i_blk.append(Assign(s_acc, zero_c))
  k_lp = Var("fbk", I)
  k_blk: list[Stmt] = []
  qv = _inline_stage(q_st, (i_lp, k_lp), k_blk)
  kv = _inline_stage(k_st, (j_lp, k_lp), k_blk)
  k_blk.append(Assign(s_acc, ops.prim("add", (s_acc, ops.prim("mul", (qv, kv), elt)), elt)))
  i_blk.append(ForRange(k_lp, ZERO, k_dim, ONE, k_blk, Scope("loop"), simd=True, reduction=("add", [s_acc])))

  s_scaled: Expr = ops.prim("mul", (s_acc, scale_c), elt) if scale_c is not None else s_acc
  cond_v: Var | None = None
  if mask_fill is not None and (mask_buf is not None or mask_cond is not None):
    c_cl = Cloner({i_m.uid: i_lp, j_m.uid: j_lp})
    cond_e: Expr = Load(mask_buf, c_cl.expr(mask_index)) if (mask_buf is not None and mask_index is not None) else c_cl.expr(mask_cond)  # type: ignore[arg-type]
    cond_v = Var("fbm", Bool)
    i_blk.append(Assign(cond_v, cond_e))
    s_eff: Expr = ops.select(cond_v, s_scaled, cast(mask_fill, elt))
  else:
    s_eff = s_scaled

  row_i = row_major_index((*b_vars, i_lp), (*batch_shape, m_dim))
  row_j = row_major_index((*b_vars, j_lp), (*batch_shape, n_dim))
  lse_i = Load(lse_arr.buf, iadd(lse_arr.offset, row_i))
  d_i = Load(d_arr.buf, iadd(d_arr.offset, row_i))
  p_ij = Var("fbp", elt)
  i_blk.append(Assign(p_ij, ops.prim("exp", (ops.prim("sub", (s_eff, lse_i), elt),), elt)))

  # dp_ij = dot(dO[i], V[j])
  dp_acc = Var("fbdp", elt)
  i_blk.append(Assign(dp_acc, zero_c))
  d_lp = Var("fbd", I)
  d_blk: list[Stmt] = []
  dov = _inline_stage(do_st, (i_lp, d_lp), d_blk)
  vv = _inline_stage(v_st, (j_lp, d_lp), d_blk)
  d_blk.append(Assign(dp_acc, ops.prim("add", (dp_acc, ops.prim("mul", (dov, vv), elt)), elt)))
  i_blk.append(ForRange(d_lp, ZERO, d_dim, ONE, d_blk, Scope("loop"), simd=True, reduction=("add", [dp_acc])))

  ds_raw: Expr = ops.prim("mul", (p_ij, ops.prim("sub", (dp_acc, d_i), elt)), elt)
  if cond_v is not None:
    ds_raw = ops.select(cond_v, ds_raw, zero_c)
  if scale_c is not None:
    ds_raw = ops.prim("mul", (ds_raw, scale_c), elt)
  ds_ij = Var("fbds", elt)
  i_blk.append(Assign(ds_ij, ds_raw))

  # dV[j, d] += p_ij * dO[i, d]
  if active_v:
    dv_base = Var("dv_b", I)
    i_blk.append(Assign(dv_base, imul(row_j, d_dim)))
    dv_d = Var("dvd", I)
    dv_blk: list[Stmt] = []
    dov2 = _inline_stage(do_st, (i_lp, dv_d), dv_blk)
    dv_pos = iadd(dv_base, dv_d)
    dv_blk.append(Store(dv_arr.buf, dv_pos, ops.prim("add", (Load(dv_arr.buf, dv_pos), ops.prim("mul", (p_ij, dov2), elt)), elt)))
    i_blk.append(ForRange(dv_d, ZERO, d_dim, ONE, dv_blk, Scope("loop"), simd=True))

  # dK[j, k] += ds_ij * Q[i, k]  and  dQ[i, k] += ds_ij * K[j, k]
  if active_k or active_q:
    dk_base = Var("dk_b", I)
    dq_base = Var("dq_b", I)
    i_blk.append(Assign(dk_base, imul(row_j, k_dim)))
    i_blk.append(Assign(dq_base, imul(row_i, k_dim)))
    dk_k = Var("dkk", I)
    dk_blk: list[Stmt] = []
    if active_k:
      qv2 = _inline_stage(q_st, (i_lp, dk_k), dk_blk)
      dk_pos = iadd(dk_base, dk_k)
      dk_blk.append(Store(dk_arr.buf, dk_pos, ops.prim("add", (Load(dk_arr.buf, dk_pos), ops.prim("mul", (ds_ij, qv2), elt)), elt)))
    if active_q:
      kv2 = _inline_stage(k_st, (j_lp, dk_k), dk_blk)
      dq_pos = iadd(dq_base, dk_k)
      dk_blk.append(Store(dq_arr.buf, dq_pos, ops.prim("add", (Load(dq_arr.buf, dq_pos), ops.prim("mul", (ds_ij, kv2), elt)), elt)))
    i_blk.append(ForRange(dk_k, ZERO, k_dim, ONE, dk_blk, Scope("loop"), simd=True))

  j_blk.append(ForRange(i_lp, ZERO, m_dim, ONE, i_blk, Scope("loop")))
  body.append(ForRange(j_lp, ZERO, n_dim, ONE, j_blk, Scope("loop")))
  body.append(ReleaseScope(scope))

  out_bufs: dict[str, Var] = {"dq": dq_arr.buf, "dk": dk_arr.buf, "dv": dv_arr.buf}
  aux_exprs: dict[str, Expr] = {
    "lse_buf": lse_arr.buf,
    "lse_offset": lse_arr.offset,
    "d_buf": d_arr.buf,
    "d_offset": d_arr.offset,
    "active_q": Const(1 if active_q else 0, I),
    "active_k": Const(1 if active_k else 0, I),
    "active_v": Const(1 if active_v else 0, I),
  }
  plan = MonoidalTilePlan(
    kind="flash_monoid_bwd",
    elt=elt,
    batch_shape=batch_shape,
    batch_index=b_vars,
    m_dim=m_dim,
    n_dim=n_dim,
    k_dim=k_dim,
    d_dim=d_dim,
    stages={"q": q_st, "k": k_st, "v": v_st, "do": do_st},
    score_op="dot",
    score_scale=scale_c,
    mask_cond=mask_cond,
    mask_index_vars=(i_m, j_m),
    mask_buf=mask_buf,
    mask_index=mask_index,
    mask_fill=None if mask_fill is None else cast(mask_fill, elt),
    tile_false_guard=tile_false_g,
    tile_true_guard=tile_true_g,
    tile_origin_vars=tile_ov,
    out_bufs=out_bufs,
    aux_exprs=aux_exprs,
  )
  el.emit(
    IndexTile(
      shape=tile_shape_all,
      tile_shape=t_shape,
      origin=origin,
      extent=extent,
      stages=[],
      body=body,
      scope=scope,
      level="shared",
      parallel=True,
      loc=el.loc(),
      monoidal_tile=plan,
    )
  )
  el.record_store(dq_arr)
  el.record_store(dk_arr)
  el.record_store(dv_arr)

  if active_q:
    tape.add(el, q_view, dq_arr)
  if active_k:
    # kt_view has shape (*batch, K, N) = k_view^T
    perm = tuple(range(len(batch_shape))) + (len(batch_shape) + 1, len(batch_shape))
    tape.add(el, el.transpose(k_view, perm), el.transpose(dk_arr, perm))
  if active_v:
    tape.add(el, v_view, dv_arr)


def _bwd_cce(tape, el, node, g: ArrayVal) -> None:
  """Law 2 Cut Cross-Entropy rematerialized backward:
  reads ONLY (H, W, targets, m, s, g_diff) — ZERO M x V logits saved!"""
  from .autodiff import UF

  h_arr, w_arr = node.inputs
  m_out, s_out, target_arr, active_h, active_w = node.info[1]
  tape.stale(el, node, h_arr)
  tape.stale(el, node, w_arr)
  tape.stale(el, node, m_out)
  tape.stale(el, node, s_out)
  tape.stale(el, node, target_arr)

  elt: ScalarType = h_arr.elt
  m_dim, k_dim = h_arr.shape
  n_dim = w_arr.shape[1]
  if isinstance(g, ArrayVal):
    if g.buf is None or not g.contig:
      g = el.as_concrete(g)
  else:
    g = el.as_concrete(el.broadcast_to(el.as_array(g), (m_dim,)))
  if target_arr.buf is None or not target_arr.contig:
    target_arr = el.as_concrete(target_arr)

  # lse = m.reshape((M,)) + log(s)
  m_1d = el.reshape(m_out, (m_dim,), checked=False)
  lse_arr = el.as_concrete(el.apply_ufunc(UF["add"], [m_1d, el.apply_ufunc(UF["log"], [s_out])]))

  dh_arr = el.alloc_array(elt, (m_dim, k_dim), zero=True, label="dh")
  dw_arr = el.alloc_array(elt, (k_dim, n_dim), zero=True, label="dw")

  h_st = _stage_from_array(el, h_arr, (), "ch")
  w_st = _stage_from_array(el, w_arr, (), "cw", transposed_last2=True)  # indexed as (v, k) -> W[k, v]

  # CPU fallback: IndexTile over (n_dim,) sequential on outer so dH has zero races,
  # while GPU runs MonoidalTilePlan(kind="cce_bwd") with parallel vocab tiles + atomicAdd(dH)!
  origin = (Var("cbo", I),)
  extent = (Var("cbe", I),)
  body: list[Stmt] = []
  scope = Scope("loop")
  zero_c = Const(0.0, elt)
  one_c = Const(1.0, elt)

  v_lp = Var("cbv", I)
  v_blk: list[Stmt] = []
  i_lp = Var("cbi", I)
  i_blk: list[Stmt] = []

  # logit = dot(H[i], W[:, v])
  dot_acc = Var("cbdot", elt)
  i_blk.append(Assign(dot_acc, zero_c))
  k_lp = Var("cbk", I)
  k_blk: list[Stmt] = []
  hv = _inline_stage(h_st, (i_lp, k_lp), k_blk)
  wv = _inline_stage(w_st, (v_lp, k_lp), k_blk)
  k_blk.append(Assign(dot_acc, ops.prim("add", (dot_acc, ops.prim("mul", (hv, wv), elt)), elt)))
  i_blk.append(ForRange(k_lp, ZERO, k_dim, ONE, k_blk, Scope("loop"), simd=True, reduction=("add", [dot_acc])))

  lse_i = Load(lse_arr.buf, iadd(lse_arr.offset, i_lp))
  p_iv = Var("cbp", elt)
  i_blk.append(Assign(p_iv, ops.prim("exp", (ops.prim("sub", (dot_acc, lse_i), elt),), elt)))

  # target_i (handle negative index wraparound!)
  t_raw = Var("cbt_r", I)
  i_blk.append(Assign(t_raw, cast(Load(target_arr.buf, iadd(target_arr.offset, i_lp)), I)))
  t_i = Var("cbt", I)
  i_blk.append(Assign(t_i, ops.select(cmp("lt", t_raw, ZERO), iadd(t_raw, n_dim), t_raw)))
  is_tgt = cmp("eq", v_lp, t_i)
  g_i = Load(g.buf, iadd(g.offset, i_lp))
  # d/d(logit) of (picked - lse) = g_i * (1_{v == y_i} - p_iv)
  dlog = Var("cbdlog", elt)
  i_blk.append(
    Assign(dlog, ops.prim("mul", (g_i, ops.prim("sub", (ops.select(is_tgt, one_c, zero_c), p_iv), elt)), elt))
  )

  k2 = Var("cbk2", I)
  k2_blk: list[Stmt] = []
  if active_w:
    hv2 = _inline_stage(h_st, (i_lp, k2), k2_blk)
    dw_pos = iadd(imul(k2, n_dim), v_lp)
    k2_blk.append(Store(dw_arr.buf, dw_pos, ops.prim("add", (Load(dw_arr.buf, dw_pos), ops.prim("mul", (dlog, hv2), elt)), elt)))
  if active_h:
    wv2 = _inline_stage(w_st, (v_lp, k2), k2_blk)
    dh_pos = iadd(imul(i_lp, k_dim), k2)
    k2_blk.append(Store(dh_arr.buf, dh_pos, ops.prim("add", (Load(dh_arr.buf, dh_pos), ops.prim("mul", (dlog, wv2), elt)), elt)))
  i_blk.append(ForRange(k2, ZERO, k_dim, ONE, k2_blk, Scope("loop"), simd=True))

  v_blk.append(ForRange(i_lp, ZERO, m_dim, ONE, i_blk, Scope("loop")))
  body.append(ForRange(v_lp, ZERO, n_dim, ONE, v_blk, Scope("loop")))
  body.append(ReleaseScope(scope))

  plan = MonoidalTilePlan(
    kind="cce_bwd",
    elt=elt,
    batch_shape=(),
    batch_index=(),
    m_dim=m_dim,
    n_dim=n_dim,
    k_dim=k_dim,
    stages={"h": h_st, "w": w_st},
    out_bufs={"dh": dh_arr.buf, "dw": dw_arr.buf},
    aux_exprs={
      "lse_buf": lse_arr.buf,
      "lse_offset": lse_arr.offset,
      "g_buf": g.buf,
      "g_offset": g.offset,
      "target_buf": target_arr.buf,
      "target_offset": target_arr.offset,
      "active_h": Const(1 if active_h else 0, I),
      "active_w": Const(1 if active_w else 0, I),
    },
  )
  el.emit(
    IndexTile(
      shape=(n_dim,),
      tile_shape=(n_dim,),
      origin=origin,
      extent=extent,
      stages=[],
      body=body,
      scope=scope,
      level="shared",
      parallel=True,
      loc=el.loc(),
      monoidal_tile=plan,
    )
  )
  el.record_store(dh_arr)
  el.record_store(dw_arr)
  if active_h:
    tape.add(el, h_arr, dh_arr)
  if active_w:
    tape.add(el, w_arr, dw_arr)


def run_algebraic_vjp(tape, el, node, g) -> None:
  handler = node.info[0]
  handler(tape, el, node, g)


# ---------------------------------------------------------------------------
# Tape optimizer: matches & replaces algebraic subgraphs on Tape.nodes
# ---------------------------------------------------------------------------


def optimize_tape(tape, el, out) -> None:
  """Rewrite recognized algebraic subgraphs on ``tape.nodes`` into compact
  rematerialized VJPs before reverse-mode traversal (Law 2)."""
  from .autodiff import Node

  if tape.dry or el.tape is not None or not tape.nodes:
    return

  nodes = tape.nodes
  n = len(nodes)
  prod_by_id: dict[int, int] = {}
  prod_by_buf: dict[int, int] = {}
  for idx, nd in enumerate(nodes):
    if nd.out is not None:
      prod_by_id[id(nd.out)] = idx
      if isinstance(nd.out, ArrayVal) and nd.out.buf is not None:
        prod_by_buf[nd.out.buf.uid] = idx

  # Build consumer map (which node indices use each producer's output)
  consumers: dict[int, set[int]] = {i: set() for i in range(n)}
  for idx, nd in enumerate(nodes):
    for inp in nd.inputs:
      p = prod_by_id.get(id(inp))
      if p is None and isinstance(inp, ArrayVal) and inp.buf is not None:
        p = prod_by_buf.get(inp.buf.uid)
      if p is not None and p < idx:
        consumers[p].add(idx)

  out_idx = prod_by_id.get(id(out))
  replaced: set[int] = set()
  replacements: dict[int, Node] = {}  # end_idx -> replacement Node

  def _internal_ok(sub_idxs: list[int], end_idx: int) -> bool:
    sub_set = set(sub_idxs)
    if sub_set & replaced:
      return False
    for u in sub_set:
      if u == end_idx:
        continue
      if u == out_idx:
        return False
      if not (consumers[u] <= sub_set):
        return False
    return True

  # --- Pass 1: FlashAttention Rematerialized VJP (Q @ K^T -> [scale] -> [mask] -> softmax -> @ V) ---
  for idx in range(n):
    if idx in replaced:
      continue
    pv = _match_batched_matmul(el, nodes, prod_by_id, idx)
    if pv is None:
      continue
    p_cand, v_view, pv_idxs = pv
    p_node_idx = prod_by_id.get(id(p_cand))
    if p_node_idx is None and p_cand.buf is not None:
      p_node_idx = prod_by_buf.get(p_cand.buf.uid)
    if p_node_idx is None:
      continue
    sm = _match_softmax_chain(el, nodes, prod_by_id, p_node_idx)
    if sm is None:
      continue
    s_in, n_max, n_sum, _p_out, sm_ax, sm_idxs = sm
    if sm_ax != s_in.rank - 1:
      continue

    # Walk optional mask (where) and optional scale (mul) backwards from s_in to Q @ K^T
    cur_val = s_in
    cur_idx = prod_by_id.get(id(cur_val))
    extra_idxs: list[int] = []
    mask_arr: ArrayVal | None = None
    mask_fill: Expr | None = None
    scale_expr: Expr | None = None

    if cur_idx is not None and nodes[cur_idx].kind == "elementwise":
      op_m, _, _ = _probe_ew(el, nodes[cur_idx])
      if op_m == "select" and len(nodes[cur_idx].inputs) == 3:
        m_cond, m_then, m_else = nodes[cur_idx].inputs
        if isinstance(m_cond, ArrayVal) and m_cond.elt is Bool and isinstance(m_else, Expr) and not isinstance(m_else, ArrayVal):
          mask_arr = m_cond
          mask_fill = m_else
          extra_idxs.append(cur_idx)
          cur_val = m_then
          cur_idx = prod_by_id.get(id(cur_val))
          if cur_idx is None and isinstance(cur_val, ArrayVal) and cur_val.buf is not None:
            cur_idx = prod_by_buf.get(cur_val.buf.uid)

    if cur_idx is not None and nodes[cur_idx].kind == "elementwise":
      op_sc, _, _ = _probe_ew(el, nodes[cur_idx])
      if op_sc == "mul" and len(nodes[cur_idx].inputs) == 2:
        a0, a1 = nodes[cur_idx].inputs
        if isinstance(a0, ArrayVal) and isinstance(a1, Expr):
          scale_expr = a1
          extra_idxs.append(cur_idx)
          cur_val = a0
          cur_idx = prod_by_id.get(id(cur_val))
          if cur_idx is None and cur_val.buf is not None:
            cur_idx = prod_by_buf.get(cur_val.buf.uid)
        elif isinstance(a1, ArrayVal) and isinstance(a0, Expr):
          scale_expr = a0
          extra_idxs.append(cur_idx)
          cur_val = a1
          cur_idx = prod_by_id.get(id(cur_val))
          if cur_idx is None and cur_val.buf is not None:
            cur_idx = prod_by_buf.get(cur_val.buf.uid)

    if cur_idx is None:
      continue
    qk = _match_batched_matmul(el, nodes, prod_by_id, cur_idx)
    if qk is None:
      continue
    q_view, kt_view, qk_idxs = qk
    # Original tape inputs for Q, K^T, V (from the matmul/mul nodes so tape.add routes to the exact producers!)
    qk_first = nodes[qk_idxs[0]]
    pv_first = nodes[pv_idxs[0]]
    q_orig, kt_orig = qk_first.inputs[0], qk_first.inputs[1]
    v_orig = pv_first.inputs[1]
    if not (isinstance(q_orig, ArrayVal) and isinstance(kt_orig, ArrayVal) and isinstance(v_orig, ArrayVal)):
      continue
    # k_view has shape (*batch, N, K) = transpose of kt_view (*batch, K, N)
    perm = tuple(range(kt_view.rank - 2)) + (kt_view.rank - 1, kt_view.rank - 2)
    k_view = el.transpose(kt_view, perm)

    all_idxs = qk_idxs + extra_idxs + sm_idxs + pv_idxs
    if not _internal_ok(all_idxs, idx):
      continue
    # Ensure no original input was overwritten and shapes match (*batch, M, K), (*batch, N, K), (*batch, N, D)
    if q_view.rank != k_view.rank or q_view.rank != v_view.rank:
      continue
    act_q = tape.is_active(el, q_orig)
    act_k = tape.is_active(el, kt_orig)
    act_v = tape.is_active(el, v_orig)
    if not (act_q or act_k or act_v):
      continue

    o_node = nodes[idx]
    m_out = n_max.out
    l_out = n_sum.out
    # Un-keepdims m_out and l_out for clean 1-element-per-row indexing (same buffer, shape (*batch, M))
    m_flat = el.reshape(m_out, m_out.shape[:-1], checked=False)
    l_flat = el.reshape(l_out, l_out.shape[:-1], checked=False)

    rep = Node(
      "algebraic_vjp",
      (q_orig, kt_orig, v_orig),
      (
        _bwd_flash_attn,
        (q_view, k_view, v_view, o_node.out, m_flat, l_flat, scale_expr, mask_arr, mask_fill, act_q, act_k, act_v),
      ),
    )
    rep.out = o_node.out
    rep.holder = o_node.holder
    rep.tick = nodes[qk_idxs[0]].tick
    rep.origin = o_node.origin
    replaced.update(all_idxs)
    replacements[idx] = rep

  # --- Pass 2: Cut Cross-Entropy Rematerialized VJP (H @ W -> max -> z=logits-m -> exp -> sum -> log; z[arange, y] - log_s) ---
  for idx in range(n):
    if idx in replaced:
      continue
    nd = nodes[idx]
    op_sub, _, _ = _probe_ew(el, nd)
    if op_sub != "sub" or len(nd.inputs) != 2:
      continue
    picked_val, log_s_val = nd.inputs
    # Match log_s = log(sum(exp(z), axis=1, keepdims=False))
    log_idx = prod_by_id.get(id(log_s_val))
    pick_idx = prod_by_id.get(id(picked_val))
    if log_idx is None or pick_idx is None:
      continue
    pick_real = _unwrap_alias(pick_idx, nodes, prod_by_id)
    if nodes[pick_real].kind != "gather":
      continue
    op_log, _, _ = _probe_ew(el, nodes[log_idx])
    if op_log != "log" or len(nodes[log_idx].inputs) != 1:
      continue
    s_val = nodes[log_idx].inputs[0]
    s_idx = prod_by_id.get(id(s_val))
    if s_idx is None or nodes[s_idx].kind != "reduce":
      continue
    s_op, s_ax, s_kd, s_act_a, s_act_i = nodes[s_idx].info
    if s_op != "add" or s_kd or not s_act_a or s_act_i:
      continue
    e_val = nodes[s_idx].inputs[0]
    e_idx = prod_by_id.get(id(e_val))
    if e_idx is None:
      continue
    op_e, _, _ = _probe_ew(el, nodes[e_idx])
    if op_e != "exp" or len(nodes[e_idx].inputs) != 1:
      continue
    z_val = nodes[e_idx].inputs[0]
    z_idx = prod_by_id.get(id(z_val))
    if z_idx is None:
      continue
    op_z, _, _ = _probe_ew(el, nodes[z_idx])
    if op_z != "sub" or len(nodes[z_idx].inputs) != 2:
      continue
    logits_val, m_val = nodes[z_idx].inputs
    m_idx = prod_by_id.get(id(m_val))
    if m_idx is None or nodes[m_idx].kind != "reduce":
      continue
    m_op, m_ax, m_kd, m_act_a, m_act_i = nodes[m_idx].info
    if m_op != "max" or not m_kd or not m_act_a or m_act_i or not _same_val(nodes[m_idx].inputs[0], logits_val):
      continue
    if not isinstance(logits_val, ArrayVal) or logits_val.rank != 2:
      continue
    if _axis_int(el, m_ax, 2) != 1 or _axis_int(el, s_ax, 2) != 1:
      continue
    # Check gather reads z_val (possibly through an alias!) with (arange, targets)
    g_base = nodes[pick_real].inputs[0]
    g_base_idx = prod_by_id.get(id(g_base))
    if g_base_idx is None and isinstance(g_base, ArrayVal) and g_base.buf is not None:
      g_base_idx = prod_by_buf.get(g_base.buf.uid)
    if g_base_idx is None:
      continue
    if _unwrap_alias(g_base_idx, nodes, prod_by_id) != z_idx:
      continue
    g_shape, _src_index, idx_arrays = nodes[pick_real].info
    if len(g_shape) != 1 or len(idx_arrays) not in (1, 2):
      continue
    target_arr = idx_arrays[-1]
    if not isinstance(target_arr, ArrayVal) or target_arr.rank != 1:
      continue
    # Check logits_val comes from 2D matmul(H, W)
    mm_idx = prod_by_id.get(id(logits_val))
    if mm_idx is None and logits_val.buf is not None:
      mm_idx = prod_by_buf.get(logits_val.buf.uid)
    if mm_idx is None or nodes[mm_idx].kind != "matmul":
      continue
    h_arr, w_arr = nodes[mm_idx].inputs
    if not (isinstance(h_arr, ArrayVal) and isinstance(w_arr, ArrayVal) and h_arr.rank == 2 and w_arr.rank == 2):
      continue
    act_h, act_w = nodes[mm_idx].info
    if not (act_h or act_w):
      continue
    all_idxs = list({mm_idx, m_idx, z_idx, e_idx, s_idx, log_idx, g_base_idx, pick_real, pick_idx, idx})
    if not _internal_ok(all_idxs, idx):
      continue
    rep = Node(
      "algebraic_vjp",
      (h_arr, w_arr),
      (_bwd_cce, (nodes[m_idx].out, nodes[s_idx].out, target_arr, act_h, act_w)),
    )
    rep.out = nd.out
    rep.holder = nd.holder
    rep.tick = nodes[mm_idx].tick
    rep.origin = nd.origin
    replaced.update(all_idxs)
    replacements[idx] = rep

  # --- Pass 3: Standalone Softmax VJP ---
  for idx in range(n):
    if idx in replaced:
      continue
    sm = _match_softmax_chain(el, nodes, prod_by_id, idx)
    if sm is None:
      continue
    x_val, _n_max, _n_sum, p_out, ax, sm_idxs = sm
    if not _internal_ok(sm_idxs, idx):
      continue
    nd = nodes[idx]
    rep = Node("algebraic_vjp", (x_val,), (_bwd_softmax, (p_out, ax)))
    rep.out = nd.out
    rep.holder = nd.holder
    rep.tick = nodes[sm_idxs[0]].tick
    rep.origin = nd.origin
    replaced.update(sm_idxs)
    replacements[idx] = rep

  # --- Pass 4: 1-Pass LayerNorm / RMSNorm VJP ---
  for idx in range(n):
    if idx in replaced:
      continue
    nd = nodes[idx]
    op_div, _, _ = _probe_ew(el, nd)
    if op_div != "div" or len(nd.inputs) != 2:
      continue
    num_val, std_val = nd.inputs
    std_idx = prod_by_id.get(id(std_val))
    if std_idx is None:
      continue
    op_sq, _, _ = _probe_ew(el, nodes[std_idx])
    if op_sq != "sqrt" or len(nodes[std_idx].inputs) != 1:
      continue
    add_eps_val = nodes[std_idx].inputs[0]
    add_eps_idx = prod_by_id.get(id(add_eps_val))
    if add_eps_idx is None:
      continue
    op_ae, _, _ = _probe_ew(el, nodes[add_eps_idx])
    if op_ae != "add" or len(nodes[add_eps_idx].inputs) != 2:
      continue
    var_val = nodes[add_eps_idx].inputs[0] if isinstance(nodes[add_eps_idx].inputs[0], ArrayVal) else nodes[add_eps_idx].inputs[1]
    var_idx = prod_by_id.get(id(var_val))
    if var_idx is None:
      continue
    # var = sum(sq, ax, kd=True) / N (or * (1/N))
    op_vd, _, _ = _probe_ew(el, nodes[var_idx])
    if op_vd not in ("div", "mul") or len(nodes[var_idx].inputs) != 2:
      continue
    vsum_val = nodes[var_idx].inputs[0]
    vsum_idx = prod_by_id.get(id(vsum_val))
    if vsum_idx is None or nodes[vsum_idx].kind != "reduce":
      continue
    vs_op, vs_ax, vs_kd, vs_act_a, vs_act_i = nodes[vsum_idx].info
    if vs_op != "add" or not vs_kd or not vs_act_a or vs_act_i:
      continue
    sq_val = nodes[vsum_idx].inputs[0]
    sq_idx = prod_by_id.get(id(sq_val))
    if sq_idx is None:
      continue
    op_p2, _, _ = _probe_ew(el, nodes[sq_idx])
    if op_p2 == "pow2":
      base_val = nodes[sq_idx].inputs[0]
    elif op_p2 == "mul" and len(nodes[sq_idx].inputs) == 2 and _same_val(nodes[sq_idx].inputs[0], nodes[sq_idx].inputs[1]):
      base_val = nodes[sq_idx].inputs[0]
    else:
      continue

    # Case A: RMSNorm (base_val == num_val == x_val)
    if _same_val(base_val, num_val) and isinstance(num_val, ArrayVal):
      ax = _axis_int(el, vs_ax, num_val.rank)
      if ax is None:
        continue
      sub = [sq_idx, vsum_idx, var_idx, add_eps_idx, std_idx, idx]
      if not _internal_ok(sub, idx):
        continue
      rep = Node("algebraic_vjp", (num_val,), (_bwd_norm, (std_val, nd.out, ax, False)))
      rep.out = nd.out
      rep.holder = nd.holder
      rep.tick = nodes[sq_idx].tick
      rep.origin = nd.origin
      replaced.update(sub)
      replacements[idx] = rep
      continue

    # Case B: LayerNorm (num_val = x - mu, base_val = x - mu, mu = sum(x, ax, kd=True) / N)
    num_idx = prod_by_id.get(id(num_val))
    base_idx = prod_by_id.get(id(base_val))
    if num_idx is None or base_idx is None:
      continue
    op_n, _, _ = _probe_ew(el, nodes[num_idx])
    op_b, _, _ = _probe_ew(el, nodes[base_idx])
    if op_n != "sub" or op_b != "sub":
      continue
    x1, mu1 = nodes[num_idx].inputs
    x2, mu2 = nodes[base_idx].inputs
    if not (_same_val(x1, x2) and _same_val(mu1, mu2) and isinstance(x1, ArrayVal)):
      continue
    mu_idx = prod_by_id.get(id(mu1))
    if mu_idx is None:
      continue
    op_md, _, _ = _probe_ew(el, nodes[mu_idx])
    if op_md not in ("div", "mul"):
      continue
    msum_val = nodes[mu_idx].inputs[0]
    msum_idx = prod_by_id.get(id(msum_val))
    if msum_idx is None or nodes[msum_idx].kind != "reduce":
      continue
    ms_op, ms_ax, ms_kd, ms_act_a, ms_act_i = nodes[msum_idx].info
    if ms_op != "add" or not ms_kd or not ms_act_a or ms_act_i or not _same_val(nodes[msum_idx].inputs[0], x1):
      continue
    ax = _axis_int(el, vs_ax, x1.rank)
    if ax is None or _axis_int(el, ms_ax, x1.rank) != ax:
      continue
    sub = list({msum_idx, mu_idx, base_idx, num_idx, sq_idx, vsum_idx, var_idx, add_eps_idx, std_idx, idx})
    if not _internal_ok(sub, idx):
      continue
    rep = Node("algebraic_vjp", (x1,), (_bwd_norm, (std_val, nd.out, ax, True)))
    rep.out = nd.out
    rep.holder = nd.holder
    rep.tick = nodes[msum_idx].tick
    rep.origin = nd.origin
    replaced.update(sub)
    replacements[idx] = rep

  if not replacements:
    return

  new_nodes: list[Node] = []
  for idx, nd in enumerate(nodes):
    if idx in replacements:
      rep = replacements[idx]
      new_nodes.append(rep)
      if rep.out is not None and id(rep.out) in tape.delayed:
        tape.delayed[id(rep.out)] = rep
    elif idx not in replaced:
      new_nodes.append(nd)
  tape.nodes[:] = new_nodes


__all__ = ["optimize_tape", "run_algebraic_vjp"]
