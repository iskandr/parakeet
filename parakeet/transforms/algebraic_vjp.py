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

from collections.abc import Callable
from dataclasses import dataclass

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
    inner = el.apply_ufunc(
      UF["sub"], [el.apply_ufunc(UF["sub"], [g, g_mean]), el.apply_ufunc(UF["mul"], [x_hat, gx_mean])]
    )
  else:
    inner = el.apply_ufunc(UF["sub"], [g, el.apply_ufunc(UF["mul"], [x_hat, gx_mean])])
  dx = el.apply_ufunc(UF["mul"], [inv_std, inner])
  tape.add(el, x_val, dx)


def _reduce_broadcast_grad(el, full_grad: ArrayVal, target: ArrayVal) -> ArrayVal:
  """Reduce ``full_grad`` over any leading or size-1 broadcast axes to match ``target.shape``."""
  from .autodiff import unbroadcast

  return unbroadcast(el, full_grad, target)


def _stage_from_broadcast_array(
  el, arr: ArrayVal, batch_vars: tuple[Var, ...], batch_shape: tuple[Expr, ...], prefix: str
) -> TileStage:
  """Build a 2D ``TileStage`` indexed by ``(r, c)`` over ``arr`` supporting static and runtime size-1 broadcast dims."""
  r_var = Var(f"{prefix}_r", I)
  c_var = Var(f"{prefix}_c", I)
  lead_n = arr.rank - 2
  b_use = batch_vars[-lead_n:] if lead_n > 0 else ()
  b_coords: list[Expr] = []
  for ax_idx, bv in enumerate(b_use):
    dim = arr.shape[ax_idx]
    if isinstance(dim, Const) and int(dim.value) == 1:
      b_coords.append(ZERO)
    elif isinstance(dim, Const) and int(dim.value) > 1:
      b_coords.append(bv)
    else:
      b_coords.append(ops.select(cmp("eq", dim, ONE), ZERO, bv))
  r_dim = arr.shape[-2]
  c_dim = arr.shape[-1]
  if isinstance(r_dim, Const) and int(r_dim.value) == 1:
    r_coord: Expr = ZERO
  elif isinstance(r_dim, Const) and int(r_dim.value) > 1:
    r_coord = r_var
  else:
    r_coord = ops.select(cmp("eq", r_dim, ONE), ZERO, r_var)
  if isinstance(c_dim, Const) and int(c_dim.value) == 1:
    c_coord: Expr = ZERO
  elif isinstance(c_dim, Const) and int(c_dim.value) > 1:
    c_coord = c_var
  else:
    c_coord = ops.select(cmp("eq", c_dim, ONE), ZERO, c_var)
  coords = (*b_coords, r_coord, c_coord)
  st_body: list[Stmt] = []
  st_scope = Scope("stage")
  with el.in_block(st_body, st_scope, loop=True):
    val = el.elem(arr, coords)
  return TileStage(
    Var(f"st_{prefix}", BufType(arr.elt)), (arr.shape[-2], arr.shape[-1]), (r_var, c_var), st_body, val, st_scope
  )


def _bwd_flash_attn(tape, el, node, g: ArrayVal) -> None:
  """Law 2 FlashAttention rematerialized backward (D_i form):
  reads ONLY (Q, K, V[, B], O, m, l, dO) — ZERO N x N tensors saved!"""
  from ..lib.numpy_lib import _reduce_prim as reduce_plain
  from .autodiff import UF

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
    bias_arr,
    bias_sign,
    active_b,
  ) = node.info[1]
  tape.stale(el, node, q_view)
  tape.stale(el, node, k_view)
  tape.stale(el, node, v_view)
  tape.stale(el, node, o_out)
  tape.stale(el, node, m_out)
  tape.stale(el, node, l_out)
  if mask_arr is not None:
    tape.stale(el, node, mask_arr)
  if bias_arr is not None:
    tape.stale(el, node, bias_arr)

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
  # Note: lse is only needed as a primal sufficient statistic (HVP computes dlse directly
  # in O(T) memory without keeping dm/dl alive from forward!).
  if el.jvp is not None and not el.jvp.muted:
    with el.jvp.computing(el):
      lse_arr = el.as_concrete(el.apply_ufunc(UF["add"], [m_out, el.apply_ufunc(UF["log"], [l_out])]))
  else:
    lse_arr = el.as_concrete(el.apply_ufunc(UF["add"], [m_out, el.apply_ufunc(UF["log"], [l_out])]))
  d_arr = el.as_concrete(
    reduce_plain(
      el, "add", el.apply_ufunc(UF["mul"], [g, o_out]), Const(o_out.rank - 1, I), Const(0, elt), "sum", False
    )
  )

  # Allocate dense zero-initialized gradients dQ, dK, dV (and optional dB)
  dq_arr = el.alloc_array(elt, (*batch_shape, m_dim, k_dim), zero=True, label="dq")
  dk_arr = el.alloc_array(elt, (*batch_shape, n_dim, k_dim), zero=True, label="dk")
  dv_arr = el.alloc_array(elt, (*batch_shape, n_dim, d_dim), zero=True, label="dv")
  db_arr = el.alloc_array(elt, (*batch_shape, m_dim, n_dim), zero=True, label="db") if active_b else None

  # Build IndexTile parallel over (*batch_shape,) on CPU (zero data races!),
  # with MonoidalTilePlan(kind="flash_monoid_bwd") for SRAM-tiled GPU execution!
  tile_shape_all = (*batch_shape, n_dim) if batch_shape else (n_dim,)
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
  bias_st = _stage_from_broadcast_array(el, bias_arr, b_vars, batch_shape, "bbias") if bias_arr is not None else None

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
  bias_sign_c = Const(float(bias_sign), elt)

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
  if bias_st is not None:
    bv = cast(_inline_stage(bias_st, (i_lp, j_lp), i_blk), elt)
    s_scaled = ops.prim("add", (s_scaled, bv if bias_sign > 0 else ops.prim("neg", (bv,), elt)), elt)
  cond_v: Var | None = None
  if mask_fill is not None and (mask_buf is not None or mask_cond is not None):
    c_cl = Cloner({i_m.uid: i_lp, j_m.uid: j_lp})
    cond_e: Expr = (
      Load(mask_buf, c_cl.expr(mask_index))
      if (mask_buf is not None and mask_index is not None)
      else c_cl.expr(mask_cond)
    )  # type: ignore[arg-type]
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

  ds_unscaled: Expr = ops.prim("mul", (p_ij, ops.prim("sub", (dp_acc, d_i), elt)), elt)
  if cond_v is not None:
    ds_unscaled = ops.select(cond_v, ds_unscaled, zero_c)
  ds_unscaled_v = Var("fbds_u", elt)
  i_blk.append(Assign(ds_unscaled_v, ds_unscaled))
  if db_arr is not None:
    db_pos = iadd(imul(row_i, n_dim), j_lp)
    db_val = ds_unscaled_v if bias_sign > 0 else ops.prim("neg", (ds_unscaled_v,), elt)
    i_blk.append(Store(db_arr.buf, db_pos, db_val))
  ds_raw: Expr = ops.prim("mul", (ds_unscaled_v, scale_c), elt) if scale_c is not None else ds_unscaled_v
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
    dv_blk.append(
      Store(dv_arr.buf, dv_pos, ops.prim("add", (Load(dv_arr.buf, dv_pos), ops.prim("mul", (p_ij, dov2), elt)), elt))
    )
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
      dk_blk.append(
        Store(dk_arr.buf, dk_pos, ops.prim("add", (Load(dk_arr.buf, dk_pos), ops.prim("mul", (ds_ij, qv2), elt)), elt))
      )
    if active_q:
      kv2 = _inline_stage(k_st, (j_lp, dk_k), dk_blk)
      dq_pos = iadd(dq_base, dk_k)
      dk_blk.append(
        Store(dq_arr.buf, dq_pos, ops.prim("add", (Load(dq_arr.buf, dq_pos), ops.prim("mul", (ds_ij, kv2), elt)), elt))
      )
    i_blk.append(ForRange(dk_k, ZERO, k_dim, ONE, dk_blk, Scope("loop"), simd=True))

  j_blk.append(ForRange(i_lp, ZERO, m_dim, ONE, i_blk, Scope("loop")))
  body.append(ForRange(j_lp, ZERO, n_dim, ONE, j_blk, Scope("loop")))
  body.append(ReleaseScope(scope))

  out_bufs: dict[str, Var] = {"dq": dq_arr.buf, "dk": dk_arr.buf, "dv": dv_arr.buf}
  if db_arr is not None:
    out_bufs["db"] = db_arr.buf
  stages: dict[str, TileStage] = {"q": q_st, "k": k_st, "v": v_st, "do": do_st}
  if bias_st is not None:
    stages["bias"] = bias_st
  aux_exprs: dict[str, Expr] = {
    "lse_buf": lse_arr.buf,
    "lse_offset": lse_arr.offset,
    "d_buf": d_arr.buf,
    "d_offset": d_arr.offset,
    "active_q": Const(1 if active_q else 0, I),
    "active_k": Const(1 if active_k else 0, I),
    "active_v": Const(1 if active_v else 0, I),
    "active_b": Const(1 if active_b else 0, I),
    "bias_sign": bias_sign_c,
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
    stages=stages,
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
  if db_arr is not None:
    el.record_store(db_arr)

  # --- Zero-O(T^2) Forward-over-Reverse HVP tangent pass when el.jvp is active ---
  # Run BEFORE tape.add() so transposed/accumulated views (like k_view^T) inherit ddk_arr!
  jvp = el.jvp
  if jvp is not None and not jvp.muted and bias_st is None:
    tq_v = jvp.tangent_array(el, q_view)
    tk_v = jvp.tangent_array(el, k_view)
    tv_v = jvp.tangent_array(el, v_view)
    tdo_v = jvp.tangent_array(el, g)
    td_v = jvp.tangent_array(el, d_arr)
    if any(x is not None for x in (tq_v, tk_v, tv_v, tdo_v, td_v)):
      with jvp.computing(el):
        _emit_flash_attn_hvp_tangent(
          el,
          jvp,
          q_view=q_view,
          k_view=k_view,
          v_view=v_view,
          g_arr=g,
          lse_arr=lse_arr,
          d_arr=d_arr,
          dq_arr=dq_arr,
          dk_arr=dk_arr,
          dv_arr=dv_arr,
          tq_v=tq_v,
          tk_v=tk_v,
          tv_v=tv_v,
          tdo_v=tdo_v,
          td_v=td_v,
          scale_c=scale_c,
          mask_cond=mask_cond,
          mask_index_vars=(i_m, j_m),
          mask_buf=mask_buf,
          mask_index=mask_index,
          mask_fill=mask_fill,
          batch_shape=batch_shape,
          m_dim=m_dim,
          n_dim=n_dim,
          k_dim=k_dim,
          d_dim=d_dim,
          elt=elt,
          active_q=active_q,
          active_k=active_k,
          active_v=active_v,
        )

  if active_q:
    tape.add(el, q_view, dq_arr)
  if active_k:
    perm = tuple(range(len(batch_shape))) + (len(batch_shape) + 1, len(batch_shape))
    tape.add(el, el.transpose(k_view, perm), el.transpose(dk_arr, perm))
  if active_v:
    tape.add(el, v_view, dv_arr)
  if active_b and db_arr is not None and bias_arr is not None:
    tape.add(el, bias_arr, _reduce_broadcast_grad(el, db_arr, bias_arr))


def _emit_flash_attn_hvp_tangent(
  el,
  jvp,
  *,
  q_view: ArrayVal,
  k_view: ArrayVal,
  v_view: ArrayVal,
  g_arr: ArrayVal,
  lse_arr: ArrayVal,
  d_arr: ArrayVal,
  dq_arr: ArrayVal,
  dk_arr: ArrayVal,
  dv_arr: ArrayVal,
  tq_v: ArrayVal | None,
  tk_v: ArrayVal | None,
  tv_v: ArrayVal | None,
  tdo_v: ArrayVal | None,
  td_v: ArrayVal | None,
  scale_c: Expr | None,
  mask_cond: Expr | None,
  mask_index_vars: tuple[Var, Var],
  mask_buf: Var | None,
  mask_index: Expr | None,
  mask_fill: Expr | None,
  batch_shape: tuple[Expr, ...],
  m_dim: Expr,
  n_dim: Expr,
  k_dim: Expr,
  d_dim: Expr,
  elt: ScalarType,
  active_q: bool,
  active_k: bool,
  active_v: bool,
) -> None:
  """Zero-O(T^2) data-parallel HVP tangent pass for rematerialized FlashAttention backward."""
  zero_c = Const(0.0, elt)
  i_m, j_m = mask_index_vars
  q_c = el.as_concrete(q_view) if q_view.buf is None else q_view
  k_c = el.as_concrete(k_view) if k_view.buf is None else k_view
  v_c = el.as_concrete(v_view) if v_view.buf is None else v_view
  g_c = el.as_concrete(g_arr) if g_arr.buf is None else g_arr
  tq_c = el.as_concrete(el.cast_array(tq_v, elt)) if tq_v is not None else None
  tk_c = el.as_concrete(el.cast_array(tk_v, elt)) if tk_v is not None else None
  tv_c = el.as_concrete(el.cast_array(tv_v, elt)) if tv_v is not None else None
  tdo_c = el.as_concrete(el.cast_array(tdo_v, elt)) if tdo_v is not None else None
  td_c = el.as_concrete(el.cast_array(td_v, elt)) if td_v is not None else None

  def _score_and_dscore(b_idx, i_e, j_e):
    s_acc = Var("hs", elt)
    ds_acc = Var("hds", elt)
    el.mutables.update((s_acc.uid, ds_acc.uid))
    el.emit(Assign(s_acc, zero_c))
    el.emit(Assign(ds_acc, zero_c))

    def k_step(kv: Expr) -> None:
      qv = el.tmp(el.elem(q_c, (*b_idx, i_e, kv)), "hq")
      kkv = el.tmp(el.elem(k_c, (*b_idx, j_e, kv)), "hk")
      el.emit(Assign(s_acc, ops.prim("add", (s_acc, ops.prim("mul", (qv, kkv), elt)), elt)))
      t_sum: Expr | None = None
      if tq_c is not None:
        tqv = el.tmp(el.elem(tq_c, (*b_idx, i_e, kv)), "htq")
        t_sum = ops.prim("mul", (tqv, kkv), elt)
      if tk_c is not None:
        tkv = el.tmp(el.elem(tk_c, (*b_idx, j_e, kv)), "htk")
        term = ops.prim("mul", (qv, tkv), elt)
        t_sum = term if t_sum is None else ops.prim("add", (t_sum, term), elt)
      if t_sum is not None:
        el.emit(Assign(ds_acc, ops.prim("add", (ds_acc, t_sum), elt)))

    el.for_range(ZERO, k_dim, k_step, name="hk")
    s_val: Expr = ops.prim("mul", (s_acc, scale_c), elt) if scale_c is not None else s_acc
    ds_val: Expr = ops.prim("mul", (ds_acc, scale_c), elt) if scale_c is not None else ds_acc
    cond_v: Expr | None = None
    if mask_fill is not None and (mask_buf is not None or mask_cond is not None):
      c_cl = Cloner({i_m.uid: i_e, j_m.uid: j_e})
      cond_v = el.tmp(
        Load(mask_buf, c_cl.expr(mask_index))
        if (mask_buf is not None and mask_index is not None)
        else c_cl.expr(mask_cond),  # type: ignore[arg-type]
        "hm",
      )
      s_val = ops.select(cond_v, s_val, cast(mask_fill, elt))
      ds_val = ops.select(cond_v, ds_val, zero_c)
    return el.tmp(s_val, "hs_v"), el.tmp(ds_val, "hds_v"), cond_v

  # 1. O(T) row-wise dlse[b, i] = sum_j P_{ij} * dS_{ij}
  dlse_arr = el.alloc_array(elt, (*batch_shape, m_dim), zero=False, label="fhvp_dlse")

  def dlse_body(idx):
    b_idx, i_e = idx[:-1], idx[-1]
    lse_i = el.tmp(el.elem(lse_arr, (*b_idx, i_e)), "lsei")
    acc = Var("hdlse", elt)
    el.mutables.add(acc.uid)
    el.emit(Assign(acc, zero_c))

    def j_step(jv: Expr) -> None:
      s_v, ds_v, _ = _score_and_dscore(b_idx, i_e, jv)
      p_ij = el.tmp(ops.prim("exp", (ops.prim("sub", (s_v, lse_i), elt),), elt), "hp")
      el.emit(Assign(acc, ops.prim("add", (acc, ops.prim("mul", (p_ij, ds_v), elt)), elt)))

    el.for_range(ZERO, n_dim, j_step, name="hj")
    el.emit(Store(dlse_arr.buf, el.index_expr(dlse_arr, idx), acc))

  el.index_map((*batch_shape, m_dim), dlse_body, ordered=False)
  el.record_store(dlse_arr)

  def _dp_and_ddp(b_idx, i_e, j_e):
    dp_a = Var("hdp", elt)
    ddp_a = Var("hddp", elt)
    el.mutables.update((dp_a.uid, ddp_a.uid))
    el.emit(Assign(dp_a, zero_c))
    el.emit(Assign(ddp_a, zero_c))

    def d_step(dv: Expr) -> None:
      dov = el.tmp(el.elem(g_c, (*b_idx, i_e, dv)), "hdo")
      vv = el.tmp(el.elem(v_c, (*b_idx, j_e, dv)), "hv")
      el.emit(Assign(dp_a, ops.prim("add", (dp_a, ops.prim("mul", (dov, vv), elt)), elt)))
      t_sum: Expr | None = None
      if tdo_c is not None:
        tdov = el.tmp(el.elem(tdo_c, (*b_idx, i_e, dv)), "htdo")
        t_sum = ops.prim("mul", (tdov, vv), elt)
      if tv_c is not None:
        tvv = el.tmp(el.elem(tv_c, (*b_idx, j_e, dv)), "htv")
        term = ops.prim("mul", (dov, tvv), elt)
        t_sum = term if t_sum is None else ops.prim("add", (t_sum, term), elt)
      if t_sum is not None:
        el.emit(Assign(ddp_a, ops.prim("add", (ddp_a, t_sum), elt)))

    el.for_range(ZERO, d_dim, d_step, name="hd")
    return dp_a, ddp_a

  # 2. Race-free parallel ddQ[b, i, k] over (*batch, M)
  if active_q:
    ddq_arr = el.alloc_array(elt, (*batch_shape, m_dim, k_dim), zero=True, label="fhvp_ddq")

    def ddq_body(idx):
      b_idx, i_e = idx[:-1], idx[-1]
      lse_i = el.tmp(el.elem(lse_arr, (*b_idx, i_e)), "lsei")
      dlse_i = el.tmp(el.elem(dlse_arr, (*b_idx, i_e)), "dlsei")
      d_i = el.tmp(el.elem(d_arr, (*b_idx, i_e)), "di")
      td_i = el.tmp(el.elem(td_c, (*b_idx, i_e)), "tdi") if td_c is not None else zero_c

      def j_step(jv: Expr) -> None:
        s_v, ds_v, cond_v = _score_and_dscore(b_idx, i_e, jv)
        p_ij = el.tmp(ops.prim("exp", (ops.prim("sub", (s_v, lse_i), elt),), elt), "hp")
        dp_t = el.tmp(ops.prim("mul", (p_ij, ops.prim("sub", (ds_v, dlse_i), elt)), elt), "hdpt")
        if cond_v is not None:
          dp_t = el.tmp(ops.select(cond_v, dp_t, zero_c), "hdpt")
        dp_ij, ddp_ij = _dp_and_ddp(b_idx, i_e, jv)
        diff = el.tmp(ops.prim("sub", (dp_ij, d_i), elt), "hdiff")
        ddiff = el.tmp(ops.prim("sub", (ddp_ij, td_i), elt), "hddiff")
        ds_u = el.tmp(ops.prim("mul", (p_ij, diff), elt), "hdsu")
        dds_u = el.tmp(
          ops.prim("add", (ops.prim("mul", (dp_t, diff), elt), ops.prim("mul", (p_ij, ddiff), elt)), elt),
          "hddsu",
        )
        if cond_v is not None:
          ds_u = el.tmp(ops.select(cond_v, ds_u, zero_c), "hdsu")
          dds_u = el.tmp(ops.select(cond_v, dds_u, zero_c), "hddsu")
        ds_sc = el.tmp(ops.prim("mul", (ds_u, scale_c), elt), "hdss") if scale_c is not None else ds_u
        dds_sc = el.tmp(ops.prim("mul", (dds_u, scale_c), elt), "hddss") if scale_c is not None else dds_u

        def k_acc(kv: Expr) -> None:
          pos = el.tmp(el.index_expr(ddq_arr, (*b_idx, i_e, kv)), "qpos")
          kkv = el.tmp(el.elem(k_c, (*b_idx, jv, kv)), "hk")
          inc = ops.prim("mul", (dds_sc, kkv), elt)
          if tk_c is not None:
            tkv = el.tmp(el.elem(tk_c, (*b_idx, jv, kv)), "htk")
            inc = ops.prim("add", (inc, ops.prim("mul", (ds_sc, tkv), elt)), elt)
          el.emit(Store(ddq_arr.buf, pos, ops.prim("add", (Load(ddq_arr.buf, pos), inc), elt)))

        el.for_range(ZERO, k_dim, k_acc, name="hqk")

      el.for_range(ZERO, n_dim, j_step, name="hqj")

    el.index_map((*batch_shape, m_dim), ddq_body, ordered=False)
    el.record_store(ddq_arr)
    jvp.bind_array(el, dq_arr, ddq_arr)

  # 3. Race-free parallel ddK[b, j, k] and ddV[b, j, d] over (*batch, N)
  if active_k or active_v:
    ddk_arr = el.alloc_array(elt, (*batch_shape, n_dim, k_dim), zero=True, label="fhvp_ddk") if active_k else None
    ddv_arr = el.alloc_array(elt, (*batch_shape, n_dim, d_dim), zero=True, label="fhvp_ddv") if active_v else None

    def ddkv_body(idx):
      b_idx, j_e = idx[:-1], idx[-1]

      def i_step(iv: Expr) -> None:
        lse_i = el.tmp(el.elem(lse_arr, (*b_idx, iv)), "lsei")
        dlse_i = el.tmp(el.elem(dlse_arr, (*b_idx, iv)), "dlsei")
        d_i = el.tmp(el.elem(d_arr, (*b_idx, iv)), "di")
        td_i = el.tmp(el.elem(td_c, (*b_idx, iv)), "tdi") if td_c is not None else zero_c
        s_v, ds_v, cond_v = _score_and_dscore(b_idx, iv, j_e)
        p_ij = el.tmp(ops.prim("exp", (ops.prim("sub", (s_v, lse_i), elt),), elt), "hp")
        dp_t = el.tmp(ops.prim("mul", (p_ij, ops.prim("sub", (ds_v, dlse_i), elt)), elt), "hdpt")
        if cond_v is not None:
          dp_t = el.tmp(ops.select(cond_v, dp_t, zero_c), "hdpt")

        if ddv_arr is not None:

          def d_acc(dv: Expr) -> None:
            pos = el.tmp(el.index_expr(ddv_arr, (*b_idx, j_e, dv)), "vpos")
            dov = el.tmp(el.elem(g_c, (*b_idx, iv, dv)), "hdo")
            inc = ops.prim("mul", (dp_t, dov), elt)
            if tdo_c is not None:
              tdov = el.tmp(el.elem(tdo_c, (*b_idx, iv, dv)), "htdo")
              inc = ops.prim("add", (inc, ops.prim("mul", (p_ij, tdov), elt)), elt)
            el.emit(Store(ddv_arr.buf, pos, ops.prim("add", (Load(ddv_arr.buf, pos), inc), elt)))

          el.for_range(ZERO, d_dim, d_acc, name="hvd")

        if ddk_arr is not None:
          dp_ij, ddp_ij = _dp_and_ddp(b_idx, iv, j_e)
          diff = el.tmp(ops.prim("sub", (dp_ij, d_i), elt), "hdiff")
          ddiff = el.tmp(ops.prim("sub", (ddp_ij, td_i), elt), "hddiff")
          ds_u = el.tmp(ops.prim("mul", (p_ij, diff), elt), "hdsu")
          dds_u = el.tmp(
            ops.prim("add", (ops.prim("mul", (dp_t, diff), elt), ops.prim("mul", (p_ij, ddiff), elt)), elt),
            "hddsu",
          )
          if cond_v is not None:
            ds_u = el.tmp(ops.select(cond_v, ds_u, zero_c), "hdsu")
            dds_u = el.tmp(ops.select(cond_v, dds_u, zero_c), "hddsu")
          ds_sc = el.tmp(ops.prim("mul", (ds_u, scale_c), elt), "hdss") if scale_c is not None else ds_u
          dds_sc = el.tmp(ops.prim("mul", (dds_u, scale_c), elt), "hddss") if scale_c is not None else dds_u

          def k_acc(kv: Expr) -> None:
            pos = el.tmp(el.index_expr(ddk_arr, (*b_idx, j_e, kv)), "kpos")
            qv = el.tmp(el.elem(q_c, (*b_idx, iv, kv)), "hq")
            inc = ops.prim("mul", (dds_sc, qv), elt)
            if tq_c is not None:
              tqv = el.tmp(el.elem(tq_c, (*b_idx, iv, kv)), "htq")
              inc = ops.prim("add", (inc, ops.prim("mul", (ds_sc, tqv), elt)), elt)
            el.emit(Store(ddk_arr.buf, pos, ops.prim("add", (Load(ddk_arr.buf, pos), inc), elt)))

          el.for_range(ZERO, k_dim, k_acc, name="hkk")

      el.for_range(ZERO, m_dim, i_step, name="hki")

    el.index_map((*batch_shape, n_dim), ddkv_body, ordered=False)
    if ddk_arr is not None:
      el.record_store(ddk_arr)
      jvp.bind_array(el, dk_arr, ddk_arr)
    if ddv_arr is not None:
      el.record_store(ddv_arr)
      jvp.bind_array(el, dv_arr, ddv_arr)


def emit_ipa_attention(
  el,
  q: ArrayVal,
  k: ArrayVal,
  v: ArrayVal,
  coords_q: ArrayVal,
  coords_k: ArrayVal,
  gamma: Expr,
  *,
  scale: Expr | None = None,
  bias: ArrayVal | None = None,
) -> ArrayVal:
  """Emit 1-pass Law 1 SRAM-tiled 3D distance-biased IPA attention + Law 2 AD node."""
  from ..errors import TypeInferenceError, UnsupportedError
  from .autodiff import Node

  if q.rank < 2 or k.rank != q.rank or v.rank != q.rank:
    raise el.error(TypeInferenceError, "ipa_attention() requires q, k, v with matching rank >= 2")
  if coords_q.rank < 2 or coords_k.rank < 2 or coords_q.rank > q.rank or coords_k.rank > k.rank:
    raise el.error(TypeInferenceError, "ipa_attention() coords_q/coords_k must have rank in [2, q.rank]")
  elt = q.elt.strong
  if elt.kind != "f":
    raise el.error(TypeInferenceError, "ipa_attention() requires floating-point inputs")
  batch_shape = q.shape[:-2]
  m_dim, k_dim = q.shape[-2], q.shape[-1]
  n_dim, d_dim = k.shape[-2], v.shape[-1]
  c_dim = coords_q.shape[-1]

  for bq_d, bk_d, bv_d in zip(batch_shape, k.shape[:-2], v.shape[:-2]):
    el.check(cmp("eq", bk_d, bq_d), "ValueError", "q and k batch dimensions must match")
    el.check(cmp("eq", bv_d, bq_d), "ValueError", "q and v batch dimensions must match")
  el.check(cmp("eq", k.shape[-1], k_dim), "ValueError", "q and k feature dimensions must match")
  el.check(cmp("eq", v.shape[-2], n_dim), "ValueError", "k and v sequence lengths must match")
  el.check(cmp("eq", coords_q.shape[-2], m_dim), "ValueError", "coords_q length must match q")
  el.check(cmp("eq", coords_k.shape[-2], n_dim), "ValueError", "coords_k length must match k")
  el.check(cmp("eq", coords_k.shape[-1], c_dim), "ValueError", "coords_q and coords_k coordinate dims must match")

  if k.elt.strong is not elt:
    k = el.cast_array(k, elt)
  if v.elt.strong is not elt:
    v = el.cast_array(v, elt)
  if coords_q.elt.strong is not elt:
    coords_q = el.cast_array(coords_q, elt)
  if coords_k.elt.strong is not elt:
    coords_k = el.cast_array(coords_k, elt)
  if bias is not None and bias.elt.strong is not elt:
    bias = el.cast_array(bias, elt)

  q_c = el.as_concrete(q) if q.buf is None else q
  k_c = el.as_concrete(k) if k.buf is None else k
  v_c = el.as_concrete(v) if v.buf is None else v
  xq_c = el.as_concrete(coords_q) if coords_q.buf is None else coords_q
  xk_c = el.as_concrete(coords_k) if coords_k.buf is None else coords_k
  b_c = None if bias is None else (el.as_concrete(bias) if bias.buf is None else bias)

  scale_c = (
    cast(scale, elt)
    if scale is not None
    else ops.select(
      cmp("gt", k_dim, ZERO),
      ops.prim("div", (Const(1.0, elt), ops.prim("sqrt", (cast(k_dim, elt),), elt)), elt),
      Const(0.0, elt),
    )
  )
  gamma_c = cast(gamma, elt)

  tape = el.tape
  if tape is not None and not tape.muted:
    if tape.is_active(el, gamma_c):
      raise el.error(UnsupportedError, "differentiating ipa_attention() with respect to gamma is not supported")
    if scale is not None and tape.is_active(el, scale_c):
      raise el.error(UnsupportedError, "differentiating ipa_attention() with respect to scale is not supported")
  want_ad = (
    tape is not None
    and not tape.muted
    and any(tape.is_active(el, x) for x in (q, k, v, coords_q, coords_k, *((bias,) if bias is not None else ())))
  )

  o_arr = el.alloc_array(elt, (*batch_shape, m_dim, d_dim), zero=False, label="ipa_o")
  lse_arr = el.alloc_array(elt, (*batch_shape, m_dim), zero=False, label="ipa_lse") if want_ad else None

  shape = (*batch_shape, m_dim)
  tile_shape: tuple[Expr | None, ...] = (ONE,) * len(shape)
  origin = tuple(Var("io", I) for _ in shape)
  extent = tuple(Var("ie", I) for _ in shape)
  b_vars = origin[:-1]
  i_var = origin[-1]

  q_st = _stage_from_array(el, q_c, b_vars, "iq")
  k_st = _stage_from_array(el, k_c, b_vars, "ik")
  v_st = _stage_from_array(el, v_c, b_vars, "iv")
  xq_st = _stage_from_broadcast_array(el, xq_c, b_vars, batch_shape, "ixq")
  xk_st = _stage_from_broadcast_array(el, xk_c, b_vars, batch_shape, "ixk")
  bias_st = _stage_from_broadcast_array(el, b_c, b_vars, batch_shape, "ibias") if b_c is not None else None

  body: list[Stmt] = []
  scope = Scope("loop")
  row_flat = row_major_index((*b_vars, i_var), shape)
  o_row_base = Var("io_base", I)
  body.append(Assign(o_row_base, imul(row_flat, d_dim)))
  zero_c = Const(0.0, elt)
  one_c = Const(1.0, elt)
  d0 = Var("id0", I)
  body.append(
    ForRange(d0, ZERO, d_dim, ONE, [Store(o_arr.buf, iadd(o_row_base, d0), zero_c)], Scope("loop"), simd=True)
  )

  m_acc = Var("im_acc", elt)
  l_acc = Var("il_acc", elt)
  body.append(Assign(m_acc, Const(float("-inf"), elt)))
  body.append(Assign(l_acc, zero_c))

  j_lp = Var("ij", I)
  j_blk: list[Stmt] = []
  k_lp = Var("ik", I)
  s_acc = Var("is_acc", elt)
  j_blk.append(Assign(s_acc, zero_c))
  k_blk: list[Stmt] = []
  qv = _inline_stage(q_st, (i_var, k_lp), k_blk)
  kv = _inline_stage(k_st, (j_lp, k_lp), k_blk)
  k_blk.append(Assign(s_acc, ops.prim("add", (s_acc, ops.prim("mul", (qv, kv), elt)), elt)))
  j_blk.append(ForRange(k_lp, ZERO, k_dim, ONE, k_blk, Scope("loop"), simd=True, reduction=("add", [s_acc])))

  c_lp = Var("ic", I)
  sq_acc = Var("isq_acc", elt)
  j_blk.append(Assign(sq_acc, zero_c))
  c_blk: list[Stmt] = []
  xqv = _inline_stage(xq_st, (i_var, c_lp), c_blk)
  xkv = _inline_stage(xk_st, (j_lp, c_lp), c_blk)
  cdf = Var("icdf", elt)
  c_blk.append(Assign(cdf, ops.prim("sub", (xqv, xkv), elt)))
  c_blk.append(Assign(sq_acc, ops.prim("add", (sq_acc, ops.prim("mul", (cdf, cdf), elt)), elt)))
  j_blk.append(ForRange(c_lp, ZERO, c_dim, ONE, c_blk, Scope("loop"), simd=True, reduction=("add", [sq_acc])))

  s_val: Expr = ops.prim("sub", (ops.prim("mul", (s_acc, scale_c), elt), ops.prim("mul", (gamma_c, sq_acc), elt)), elt)
  if bias_st is not None:
    bv = cast(_inline_stage(bias_st, (i_var, j_lp), j_blk), elt)
    s_val = ops.prim("add", (s_val, bv), elt)
  s_cur = Var("is_cur", elt)
  j_blk.append(Assign(s_cur, s_val))

  m_new = Var("im_new", elt)
  alpha = Var("ialpha", elt)
  beta = Var("ibeta", elt)
  is_first = cmp("eq", j_lp, ZERO)
  j_blk.append(Assign(m_new, ops.select(is_first, s_cur, ops.prim("max", (m_acc, s_cur), elt))))
  j_blk.append(
    Assign(alpha, ops.select(is_first, zero_c, ops.prim("exp", (ops.prim("sub", (m_acc, m_new), elt),), elt)))
  )
  j_blk.append(Assign(beta, ops.select(is_first, one_c, ops.prim("exp", (ops.prim("sub", (s_cur, m_new), elt),), elt))))
  j_blk.append(Assign(l_acc, ops.prim("add", (ops.prim("mul", (l_acc, alpha), elt), beta), elt)))
  j_blk.append(Assign(m_acc, m_new))

  d_lp = Var("id", I)
  d_blk: list[Stmt] = []
  vv = _inline_stage(v_st, (j_lp, d_lp), d_blk)
  o_pos = iadd(o_row_base, d_lp)
  d_blk.append(
    Store(
      o_arr.buf,
      o_pos,
      ops.prim("add", (ops.prim("mul", (Load(o_arr.buf, o_pos), alpha), elt), ops.prim("mul", (beta, vv), elt)), elt),
    )
  )
  j_blk.append(ForRange(d_lp, ZERO, d_dim, ONE, d_blk, Scope("loop"), simd=True))
  body.append(ForRange(j_lp, ZERO, n_dim, ONE, j_blk, Scope("loop")))

  inv_l = Var("iinv_l", elt)
  has_any = cmp("gt", l_acc, zero_c)
  body.append(Assign(inv_l, ops.select(has_any, ops.prim("div", (one_c, l_acc), elt), zero_c)))
  d_end = Var("id_end", I)
  end_pos = iadd(o_row_base, d_end)
  body.append(
    ForRange(
      d_end,
      ZERO,
      d_dim,
      ONE,
      [Store(o_arr.buf, end_pos, ops.prim("mul", (Load(o_arr.buf, end_pos), inv_l), elt))],
      Scope("loop"),
      simd=True,
    )
  )
  out_bufs: dict[str, Var] = {"o": o_arr.buf}
  if lse_arr is not None:
    lse_val = ops.select(
      has_any, ops.prim("add", (m_acc, ops.prim("log", (l_acc,), elt)), elt), Const(float("-inf"), elt)
    )
    body.append(Store(lse_arr.buf, row_flat, lse_val))
    out_bufs["lse"] = lse_arr.buf
  body.append(ReleaseScope(scope))

  stages: dict[str, TileStage] = {"q": q_st, "k": k_st, "v": v_st, "xq": xq_st, "xk": xk_st}
  if bias_st is not None:
    stages["bias"] = bias_st
  plan = MonoidalTilePlan(
    kind="flash_monoid",
    elt=elt,
    batch_shape=batch_shape,
    batch_index=b_vars,
    m_dim=m_dim,
    n_dim=n_dim,
    k_dim=k_dim,
    d_dim=d_dim,
    stages=stages,
    score_op="dot",
    score_scale=scale_c,
    out_bufs=out_bufs,
    aux_exprs={"ipa_gamma": gamma_c, "ipa_c_dim": c_dim},
    split_k=False,
  )
  el.emit(
    IndexTile(
      shape=shape,
      tile_shape=tile_shape,
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
  el.record_store(o_arr)
  if lse_arr is not None:
    el.record_store(lse_arr)

  if want_ad and tape is not None and lse_arr is not None:
    act_q = tape.is_active(el, q)
    act_k = tape.is_active(el, k)
    act_v = tape.is_active(el, v)
    act_xq = tape.is_active(el, coords_q)
    act_xk = tape.is_active(el, coords_k)
    act_b = bias is not None and tape.is_active(el, bias)
    inps = (q, k, v, coords_q, coords_k) + ((bias,) if bias is not None else ())
    node = Node(
      "algebraic_vjp",
      inps,
      (
        _bwd_ipa_attn,
        (
          q,
          k,
          v,
          coords_q,
          coords_k,
          bias,
          q_c,
          k_c,
          v_c,
          xq_c,
          xk_c,
          b_c,
          o_arr,
          lse_arr,
          scale_c,
          gamma_c,
          act_q,
          act_k,
          act_v,
          act_xq,
          act_xk,
          act_b,
        ),
      ),
    )
    tape.record(el, node)
    tape.register(el, node, o_arr)
  return o_arr


def _bwd_ipa_attn(tape, el, node, g: ArrayVal) -> None:
  """Law 2 D_i-recomputed backward for 3D distance-biased IPA attention (ZERO N x N saved)."""
  from ..lib.numpy_lib import _reduce_prim as reduce_plain
  from .autodiff import UF

  (
    q_orig,
    k_orig,
    v_orig,
    xq_orig,
    xk_orig,
    b_orig,
    q_c,
    k_c,
    v_c,
    xq_c,
    xk_c,
    b_c,
    o_out,
    lse_arr,
    scale_c,
    gamma_c,
    act_q,
    act_k,
    act_v,
    act_xq,
    act_xk,
    act_b,
  ) = node.info[1]
  for arr in (q_c, k_c, v_c, xq_c, xk_c, o_out, lse_arr):
    tape.stale(el, node, arr)
  if b_c is not None:
    tape.stale(el, node, b_c)

  elt: ScalarType = q_c.elt
  if g.buf is None or not g.contig:
    g = el.as_concrete(g)
  batch_shape = q_c.shape[:-2]
  m_dim, k_dim = q_c.shape[-2], q_c.shape[-1]
  n_dim, d_dim = k_c.shape[-2], v_c.shape[-1]
  c_dim = xq_c.shape[-1]

  d_arr = el.as_concrete(
    reduce_plain(
      el, "add", el.apply_ufunc(UF["mul"], [g, o_out]), Const(o_out.rank - 1, I), Const(0, elt), "sum", False
    )
  )

  dq_arr = el.alloc_array(elt, (*batch_shape, m_dim, k_dim), zero=True, label="ipa_dq")
  dk_arr = el.alloc_array(elt, (*batch_shape, n_dim, k_dim), zero=True, label="ipa_dk")
  dv_arr = el.alloc_array(elt, (*batch_shape, n_dim, d_dim), zero=True, label="ipa_dv")
  dxq_arr = el.alloc_array(elt, (*batch_shape, m_dim, c_dim), zero=True, label="ipa_dxq") if act_xq else None
  dxk_arr = el.alloc_array(elt, (*batch_shape, n_dim, c_dim), zero=True, label="ipa_dxk") if act_xk else None
  db_arr = el.alloc_array(elt, (*batch_shape, m_dim, n_dim), zero=True, label="ipa_db") if act_b else None

  tile_shape_all = (*batch_shape, n_dim) if batch_shape else (n_dim,)
  t_shape: tuple[Expr | None, ...] = (*(ONE,) * len(batch_shape), None) if batch_shape else (n_dim,)
  origin = tuple(Var("ibo", I) for _ in tile_shape_all)
  extent = tuple(Var("ibe", I) for _ in tile_shape_all)
  b_vars = origin[:-1] if batch_shape else ()

  q_st = _stage_from_array(el, q_c, b_vars, "ibq")
  k_st = _stage_from_array(el, k_c, b_vars, "ibk")
  v_st = _stage_from_array(el, v_c, b_vars, "ibv")
  do_st = _stage_from_array(el, g, b_vars, "ibdo")
  xq_st = _stage_from_broadcast_array(el, xq_c, b_vars, batch_shape, "ibxq")
  xk_st = _stage_from_broadcast_array(el, xk_c, b_vars, batch_shape, "ibxk")
  bias_st = _stage_from_broadcast_array(el, b_c, b_vars, batch_shape, "ibbias") if b_c is not None else None

  body: list[Stmt] = []
  scope = Scope("loop")
  zero_c = Const(0.0, elt)
  neg2_gamma = ops.prim("mul", (Const(-2.0, elt), gamma_c), elt)

  j_lp = Var("ibj", I)
  j_blk: list[Stmt] = []
  i_lp = Var("ibi", I)
  i_blk: list[Stmt] = []

  s_acc = Var("ibs", elt)
  i_blk.append(Assign(s_acc, zero_c))
  k_lp = Var("ibk", I)
  k_blk: list[Stmt] = []
  qv = _inline_stage(q_st, (i_lp, k_lp), k_blk)
  kv = _inline_stage(k_st, (j_lp, k_lp), k_blk)
  k_blk.append(Assign(s_acc, ops.prim("add", (s_acc, ops.prim("mul", (qv, kv), elt)), elt)))
  i_blk.append(ForRange(k_lp, ZERO, k_dim, ONE, k_blk, Scope("loop"), simd=True, reduction=("add", [s_acc])))

  c_lp = Var("ibc", I)
  sq_acc = Var("ibsq", elt)
  i_blk.append(Assign(sq_acc, zero_c))
  c_blk: list[Stmt] = []
  xqv = _inline_stage(xq_st, (i_lp, c_lp), c_blk)
  xkv = _inline_stage(xk_st, (j_lp, c_lp), c_blk)
  cdf = Var("ibcdf", elt)
  c_blk.append(Assign(cdf, ops.prim("sub", (xqv, xkv), elt)))
  c_blk.append(Assign(sq_acc, ops.prim("add", (sq_acc, ops.prim("mul", (cdf, cdf), elt)), elt)))
  i_blk.append(ForRange(c_lp, ZERO, c_dim, ONE, c_blk, Scope("loop"), simd=True, reduction=("add", [sq_acc])))

  s_eff: Expr = ops.prim("sub", (ops.prim("mul", (s_acc, scale_c), elt), ops.prim("mul", (gamma_c, sq_acc), elt)), elt)
  if bias_st is not None:
    s_eff = ops.prim("add", (s_eff, cast(_inline_stage(bias_st, (i_lp, j_lp), i_blk), elt)), elt)

  row_i = row_major_index((*b_vars, i_lp), (*batch_shape, m_dim))
  row_j = row_major_index((*b_vars, j_lp), (*batch_shape, n_dim))
  lse_i = Load(lse_arr.buf, iadd(lse_arr.offset, row_i))
  d_i = Load(d_arr.buf, iadd(d_arr.offset, row_i))
  p_ij = Var("ibp", elt)
  i_blk.append(Assign(p_ij, ops.prim("exp", (ops.prim("sub", (s_eff, lse_i), elt),), elt)))

  dp_acc = Var("ibdp", elt)
  i_blk.append(Assign(dp_acc, zero_c))
  d_lp = Var("ibd", I)
  d_blk: list[Stmt] = []
  dov = _inline_stage(do_st, (i_lp, d_lp), d_blk)
  vv = _inline_stage(v_st, (j_lp, d_lp), d_blk)
  d_blk.append(Assign(dp_acc, ops.prim("add", (dp_acc, ops.prim("mul", (dov, vv), elt)), elt)))
  i_blk.append(ForRange(d_lp, ZERO, d_dim, ONE, d_blk, Scope("loop"), simd=True, reduction=("add", [dp_acc])))

  ds_u = Var("ibds_u", elt)
  i_blk.append(Assign(ds_u, ops.prim("mul", (p_ij, ops.prim("sub", (dp_acc, d_i), elt)), elt)))
  if db_arr is not None:
    i_blk.append(Store(db_arr.buf, iadd(imul(row_i, n_dim), j_lp), ds_u))
  ds_ij = Var("ibds", elt)
  i_blk.append(Assign(ds_ij, ops.prim("mul", (ds_u, scale_c), elt)))

  if act_v:
    dv_base = Var("idv_b", I)
    i_blk.append(Assign(dv_base, imul(row_j, d_dim)))
    dv_d = Var("idvd", I)
    dv_blk: list[Stmt] = []
    dov2 = _inline_stage(do_st, (i_lp, dv_d), dv_blk)
    dv_pos = iadd(dv_base, dv_d)
    dv_blk.append(
      Store(dv_arr.buf, dv_pos, ops.prim("add", (Load(dv_arr.buf, dv_pos), ops.prim("mul", (p_ij, dov2), elt)), elt))
    )
    i_blk.append(ForRange(dv_d, ZERO, d_dim, ONE, dv_blk, Scope("loop"), simd=True))

  if act_k or act_q:
    dk_base = Var("idk_b", I)
    dq_base = Var("idq_b", I)
    i_blk.append(Assign(dk_base, imul(row_j, k_dim)))
    i_blk.append(Assign(dq_base, imul(row_i, k_dim)))
    dk_k = Var("idkk", I)
    dk_blk: list[Stmt] = []
    if act_k:
      qv2 = _inline_stage(q_st, (i_lp, dk_k), dk_blk)
      dk_pos = iadd(dk_base, dk_k)
      dk_blk.append(
        Store(dk_arr.buf, dk_pos, ops.prim("add", (Load(dk_arr.buf, dk_pos), ops.prim("mul", (ds_ij, qv2), elt)), elt))
      )
    if act_q:
      kv2 = _inline_stage(k_st, (j_lp, dk_k), dk_blk)
      dq_pos = iadd(dq_base, dk_k)
      dk_blk.append(
        Store(dq_arr.buf, dq_pos, ops.prim("add", (Load(dq_arr.buf, dq_pos), ops.prim("mul", (ds_ij, kv2), elt)), elt))
      )
    i_blk.append(ForRange(dk_k, ZERO, k_dim, ONE, dk_blk, Scope("loop"), simd=True))

  if act_xq or act_xk:
    dc_coef = Var("idc_coef", elt)
    i_blk.append(Assign(dc_coef, ops.prim("mul", (neg2_gamma, ds_u), elt)))
    dxq_base = Var("idxq_b", I)
    dxk_base = Var("idxk_b", I)
    i_blk.append(Assign(dxq_base, imul(row_i, c_dim)))
    i_blk.append(Assign(dxk_base, imul(row_j, c_dim)))
    dc_c = Var("idcc", I)
    dc_blk: list[Stmt] = []
    xqv2 = _inline_stage(xq_st, (i_lp, dc_c), dc_blk)
    xkv2 = _inline_stage(xk_st, (j_lp, dc_c), dc_blk)
    gc = Var("ibgc", elt)
    dc_blk.append(Assign(gc, ops.prim("mul", (dc_coef, ops.prim("sub", (xqv2, xkv2), elt)), elt)))
    if dxq_arr is not None:
      pos_q = iadd(dxq_base, dc_c)
      dc_blk.append(Store(dxq_arr.buf, pos_q, ops.prim("add", (Load(dxq_arr.buf, pos_q), gc), elt)))
    if dxk_arr is not None:
      pos_k = iadd(dxk_base, dc_c)
      dc_blk.append(Store(dxk_arr.buf, pos_k, ops.prim("sub", (Load(dxk_arr.buf, pos_k), gc), elt)))
    i_blk.append(ForRange(dc_c, ZERO, c_dim, ONE, dc_blk, Scope("loop"), simd=True))

  j_blk.append(ForRange(i_lp, ZERO, m_dim, ONE, i_blk, Scope("loop")))
  body.append(ForRange(j_lp, ZERO, n_dim, ONE, j_blk, Scope("loop")))
  body.append(ReleaseScope(scope))

  out_bufs: dict[str, Var] = {"dq": dq_arr.buf, "dk": dk_arr.buf, "dv": dv_arr.buf}
  if dxq_arr is not None:
    out_bufs["dxq"] = dxq_arr.buf
  if dxk_arr is not None:
    out_bufs["dxk"] = dxk_arr.buf
  if db_arr is not None:
    out_bufs["db"] = db_arr.buf
  stages: dict[str, TileStage] = {"q": q_st, "k": k_st, "v": v_st, "do": do_st, "xq": xq_st, "xk": xk_st}
  if bias_st is not None:
    stages["bias"] = bias_st
  aux_exprs: dict[str, Expr] = {
    "lse_buf": lse_arr.buf,
    "lse_offset": lse_arr.offset,
    "d_buf": d_arr.buf,
    "d_offset": d_arr.offset,
    "active_q": Const(1 if act_q else 0, I),
    "active_k": Const(1 if act_k else 0, I),
    "active_v": Const(1 if act_v else 0, I),
    "active_b": Const(1 if act_b else 0, I),
    "bias_sign": Const(1.0, elt),
    "ipa_gamma": gamma_c,
    "ipa_c_dim": c_dim,
    "active_xq": Const(1 if act_xq else 0, I),
    "active_xk": Const(1 if act_xk else 0, I),
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
    stages=stages,
    score_op="dot",
    score_scale=scale_c,
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
  for arr in (dq_arr, dk_arr, dv_arr, dxq_arr, dxk_arr, db_arr):
    if arr is not None:
      el.record_store(arr)

  if act_q:
    tape.add(el, q_orig, dq_arr)
  if act_k:
    tape.add(el, k_orig, dk_arr)
  if act_v:
    tape.add(el, v_orig, dv_arr)
  if act_xq and dxq_arr is not None:
    tape.add(el, xq_orig, _reduce_broadcast_grad(el, dxq_arr, xq_orig))
  if act_xk and dxk_arr is not None:
    tape.add(el, xk_orig, _reduce_broadcast_grad(el, dxk_arr, xk_orig))
  if act_b and db_arr is not None and b_orig is not None:
    tape.add(el, b_orig, _reduce_broadcast_grad(el, db_arr, b_orig))


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
    k2_blk.append(
      Store(dw_arr.buf, dw_pos, ops.prim("add", (Load(dw_arr.buf, dw_pos), ops.prim("mul", (dlog, hv2), elt)), elt))
    )
  if active_h:
    wv2 = _inline_stage(w_st, (v_lp, k2), k2_blk)
    dh_pos = iadd(imul(i_lp, k_dim), k2)
    k2_blk.append(
      Store(dh_arr.buf, dh_pos, ops.prim("add", (Load(dh_arr.buf, dh_pos), ops.prim("mul", (dlog, wv2), elt)), elt))
    )
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


@dataclass(frozen=True)
class AlgebraicADRule:
  """Declarative descriptor for a Law 2 algebraic AD rematerialization rule."""

  name: str
  description: str
  vjp_handler: Callable | None = None
  jvp_supported: bool = False
  hvp_supported: bool = False


ALGEBRAIC_AD_RULES: tuple[AlgebraicADRule, ...] = (
  AlgebraicADRule(
    name="flash_attention",
    description="Zero-O(T^2) SRAM-rematerialized attention VJP, JVP, and HVP",
    vjp_handler=_bwd_flash_attn,
    jvp_supported=True,
    hvp_supported=True,
  ),
  AlgebraicADRule(
    name="cut_cross_entropy",
    description="Zero-O(M*V) SRAM-rematerialized linear cross-entropy VJP",
    vjp_handler=_bwd_cce,
    jvp_supported=False,
    hvp_supported=False,
  ),
  AlgebraicADRule(
    name="softmax",
    description="Shift-invariant closed-form softmax VJP and JVP",
    vjp_handler=_bwd_softmax,
    jvp_supported=True,
    hvp_supported=True,
  ),
  AlgebraicADRule(
    name="layernorm_rmsnorm",
    description="Single-pass closed-form LayerNorm / RMSNorm VJP, JVP, and HVP",
    vjp_handler=_bwd_norm,
    jvp_supported=True,
    hvp_supported=True,
  ),
)


def optimize_tape(tape, el, out) -> None:
  """Rewrite recognized algebraic subgraphs on ``tape.nodes`` into compact
  rematerialized VJPs before reverse-mode traversal (Law 2).

  Supports both top-level straight-line tapes and checkpointed loop-body child
  tapes, as well as multi-output trees.
  """
  from .ad_loops import flatten_tree
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

  out_leaves, _ = flatten_tree(out)
  out_idxs: set[int] = set()
  for ol in out_leaves:
    oi = prod_by_id.get(id(ol))
    if oi is None and isinstance(ol, ArrayVal) and ol.buf is not None:
      oi = prod_by_buf.get(ol.buf.uid)
    if oi is not None:
      out_idxs.add(oi)
  replaced: set[int] = set()
  replacements: dict[int, Node] = {}  # end_idx -> replacement Node

  def _internal_ok(sub_idxs: list[int], end_idx: int) -> bool:
    sub_set = set(sub_idxs)
    if sub_set & replaced:
      return False
    for u in sub_set:
      if u == end_idx:
        continue
      if u in out_idxs:
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

    # Walk optional mask (where), optional pair bias (+ B / - B), and optional scale (mul) backwards from s_in to Q @ K^T
    cur_val = s_in
    cur_idx = prod_by_id.get(id(cur_val))
    extra_idxs: list[int] = []
    mask_arr: ArrayVal | None = None
    mask_fill: Expr | None = None
    bias_arr: ArrayVal | None = None
    bias_sign: int = 1
    scale_expr: Expr | None = None

    if cur_idx is not None and nodes[cur_idx].kind == "elementwise":
      op_m, _, _ = _probe_ew(el, nodes[cur_idx])
      if op_m == "select" and len(nodes[cur_idx].inputs) == 3:
        m_cond, m_then, m_else = nodes[cur_idx].inputs
        if (
          isinstance(m_cond, ArrayVal)
          and m_cond.elt is Bool
          and isinstance(m_else, Expr)
          and not isinstance(m_else, ArrayVal)
        ):
          mask_arr = m_cond
          mask_fill = m_else
          extra_idxs.append(cur_idx)
          cur_val = m_then
          cur_idx = prod_by_id.get(id(cur_val))
          if cur_idx is None and isinstance(cur_val, ArrayVal) and cur_val.buf is not None:
            cur_idx = prod_by_buf.get(cur_val.buf.uid)

    if cur_idx is not None and nodes[cur_idx].kind == "elementwise":
      op_b, _, _ = _probe_ew(el, nodes[cur_idx])
      if op_b in ("add", "sub") and len(nodes[cur_idx].inputs) == 2:
        b0, b1 = nodes[cur_idx].inputs
        if isinstance(b0, ArrayVal) and isinstance(b1, ArrayVal) and b0.rank >= 2 and b1.rank >= 2:
          # Pick whichever side traces back to (mul? ->) batched_matmul
          def _resolves_to_qk(cand: ArrayVal) -> bool:
            ci = prod_by_id.get(id(cand))
            if ci is None and cand.buf is not None:
              ci = prod_by_buf.get(cand.buf.uid)
            if ci is None:
              return False
            if _match_batched_matmul(el, nodes, prod_by_id, ci) is not None:
              return True
            if nodes[ci].kind == "elementwise":
              op_s, _, _ = _probe_ew(el, nodes[ci])
              if op_s == "mul" and len(nodes[ci].inputs) == 2:
                s0, s1 = nodes[ci].inputs
                inner = (
                  s0
                  if isinstance(s0, ArrayVal) and isinstance(s1, Expr)
                  else (s1 if isinstance(s1, ArrayVal) and isinstance(s0, Expr) else None)
                )
                if inner is not None:
                  ii = prod_by_id.get(id(inner))
                  if ii is None and inner.buf is not None:
                    ii = prod_by_buf.get(inner.buf.uid)
                  return ii is not None and _match_batched_matmul(el, nodes, prod_by_id, ii) is not None
            return False

          if _resolves_to_qk(b0):
            bias_arr = b1
            bias_sign = 1 if op_b == "add" else -1
            extra_idxs.append(cur_idx)
            cur_val = b0
            cur_idx = prod_by_id.get(id(cur_val))
            if cur_idx is None and cur_val.buf is not None:
              cur_idx = prod_by_buf.get(cur_val.buf.uid)
          elif op_b == "add" and _resolves_to_qk(b1):
            bias_arr = b0
            bias_sign = 1
            extra_idxs.append(cur_idx)
            cur_val = b1
            cur_idx = prod_by_id.get(id(cur_val))
            if cur_idx is None and cur_val.buf is not None:
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
    qk_first = nodes[qk_idxs[0]]
    pv_first = nodes[pv_idxs[0]]
    q_orig, kt_orig = qk_first.inputs[0], qk_first.inputs[1]
    v_orig = pv_first.inputs[1]
    if not (isinstance(q_orig, ArrayVal) and isinstance(kt_orig, ArrayVal) and isinstance(v_orig, ArrayVal)):
      continue
    perm = tuple(range(kt_view.rank - 2)) + (kt_view.rank - 1, kt_view.rank - 2)
    k_view = el.transpose(kt_view, perm)

    all_idxs = qk_idxs + extra_idxs + sm_idxs + pv_idxs
    if not _internal_ok(all_idxs, idx):
      continue
    if q_view.rank != k_view.rank or q_view.rank != v_view.rank:
      continue
    if bias_arr is not None and bias_arr.rank > q_view.rank:
      continue
    act_q = tape.is_active(el, q_orig)
    act_k = tape.is_active(el, kt_orig)
    act_v = tape.is_active(el, v_orig)
    act_b = bias_arr is not None and tape.is_active(el, bias_arr)
    if not (act_q or act_k or act_v or act_b):
      continue

    o_node = nodes[idx]
    m_out = n_max.out
    l_out = n_sum.out
    m_flat = el.reshape(m_out, m_out.shape[:-1], checked=False)
    l_flat = el.reshape(l_out, l_out.shape[:-1], checked=False)

    rep_inps = (q_orig, kt_orig, v_orig) + ((bias_arr,) if bias_arr is not None else ())
    rep = Node(
      "algebraic_vjp",
      rep_inps,
      (
        _bwd_flash_attn,
        (
          q_view,
          k_view,
          v_view,
          o_node.out,
          m_flat,
          l_flat,
          scale_expr,
          mask_arr,
          mask_fill,
          act_q,
          act_k,
          act_v,
          bias_arr,
          bias_sign,
          act_b,
        ),
      ),
    )
    rep.out = o_node.out
    rep.holder = o_node.holder
    rep.tick = o_node.tick
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
    var_val = (
      nodes[add_eps_idx].inputs[0]
      if isinstance(nodes[add_eps_idx].inputs[0], ArrayVal)
      else nodes[add_eps_idx].inputs[1]
    )
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
    elif (
      op_p2 == "mul" and len(nodes[sq_idx].inputs) == 2 and _same_val(nodes[sq_idx].inputs[0], nodes[sq_idx].inputs[1])
    ):
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


__all__ = ["ALGEBRAIC_AD_RULES", "AlgebraicADRule", "emit_ipa_attention", "optimize_tape", "run_algebraic_vjp"]
