"""Tests for AD Upgrade Wave 2 (v1.3.0):
- par.jvp (forward-mode AD: scalars, arrays, matmul, reduce, scan, fold, map,
  gather, scatter_add, if/else, runtime for/while loops, Record, RaggedArray)
- @par.custom_jvp
- par.vjp (general non-scalar reverse-mode AD: arrays, tuples, records, RaggedArray)
- par.linearize & par.linear_transpose + transpose identity <JVP(x; v), u> == <v, VJP(x; u)>
- par.hvp (forward-over-reverse Hessian-vector products), par.jacfwd, par.jacrev, par.hessian
- Higher-order grad(grad(f)) through pack, cumsum, map, gather, scatter_add
- Adjoints & JVP for topk / topk_fold, sort, FrozenHashTable.get, SortedIndex.get
- np.add.at & canonical for-loop lifting to IndexScatter (including grad & jvp)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest

import parakeet as par
from parakeet import FrozenHashTable, RaggedArray, SortedIndex


class Pair(NamedTuple):
  w: np.ndarray
  b: np.ndarray
  tag: int = 0


@dataclass(frozen=True)
class Layer:
  weight: np.ndarray
  bias: np.ndarray
  scale: float
  label: str = "dense"


def _fd_jvp(fn, primals, tangents, eps=1e-6):
  p_plus = tuple(p + eps * t if isinstance(p, (np.ndarray, float)) else p for p, t in zip(primals, tangents))
  p_minus = tuple(p - eps * t if isinstance(p, (np.ndarray, float)) else p for p, t in zip(primals, tangents))
  y_plus = fn(*p_plus)
  y_minus = fn(*p_minus)
  if isinstance(y_plus, tuple):
    return tuple((a - b) / (2 * eps) for a, b in zip(y_plus, y_minus))
  return (y_plus - y_minus) / (2 * eps)


def _fd_hvp(fn, x, v, eps=1e-5):
  g = par.grad(fn)
  return (g(x + eps * v) - g(x - eps * v)) / (2 * eps)


# ---------------------------------------------------------------------------
# 1. Forward-mode JVP: scalars, arrays, ufuncs, matmul, reductions, views
# ---------------------------------------------------------------------------


def test_jvp_scalar_and_elementwise():
  def f(x, y):
    return np.sin(x * y) + np.exp(-x * 0.5) / (1.0 + y * y)

  rng = np.random.default_rng(0)
  x = rng.normal(size=(8, 6))
  y = rng.normal(size=(8, 6))
  dx = rng.normal(size=(8, 6))
  dy = rng.normal(size=(8, 6))

  out, dout = par.jvp(f, (x, y), (dx, dy))
  np.testing.assert_allclose(out, f(x, y), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(dout, _fd_jvp(f, (x, y), (dx, dy)), rtol=1e-5, atol=1e-6)


def test_jvp_matmul_and_einsum():
  def f(a, b, c):
    return (a @ b) @ c + np.sum(a * a)

  rng = np.random.default_rng(1)
  a = rng.normal(size=(5, 4))
  b = rng.normal(size=(4, 6))
  c = rng.normal(size=(6, 3))
  da = rng.normal(size=(5, 4))
  db = rng.normal(size=(4, 6))
  dc = rng.normal(size=(6, 3))

  out, dout = par.jvp(f, (a, b, c), (da, db, dc))
  np.testing.assert_allclose(out, f(a, b, c), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(dout, _fd_jvp(f, (a, b, c), (da, db, dc)), rtol=1e-5, atol=1e-6)


def test_jvp_reductions_sum_max_min_prod():
  def f_sum(x):
    return np.sum(x * x, axis=1)

  def f_max(x):
    return np.max(x, axis=1)

  def f_min(x):
    return np.min(x, axis=1)

  def f_prod(x):
    return np.prod(x, axis=1)

  rng = np.random.default_rng(2)
  x = rng.normal(size=(4, 5)) + 0.5
  dx = rng.normal(size=(4, 5))
  for fn in (f_sum, f_max, f_min, f_prod):
    out, dout = par.jvp(fn, (x,), (dx,))
    np.testing.assert_allclose(out, fn(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dout, _fd_jvp(fn, (x,), (dx,)), rtol=1e-5, atol=1e-6)

  # Prod with a single zero element per row (zero-safe product rule)
  xz = np.array([[2.0, 0.0, 3.0], [1.5, 4.0, 2.0]])
  dxz = np.array([[0.5, 1.2, -0.3], [0.1, -0.2, 0.4]])
  out_z, dout_z = par.jvp(f_prod, (xz,), (dxz,))
  np.testing.assert_allclose(out_z, np.array([0.0, 12.0]))
  np.testing.assert_allclose(dout_z, np.array([1.2 * 6.0, 0.1 * 8.0 - 0.2 * 3.0 + 0.4 * 6.0]), rtol=1e-12)

  # Max tie-averaging matches reverse-mode subgradient
  xt = np.array([3.0, 3.0, 1.0])
  dxt = np.array([2.0, 4.0, 10.0])
  out_t, dout_t = par.jvp(lambda a: np.max(a), (xt,), (dxt,))
  assert out_t == pytest.approx(3.0)
  assert dout_t == pytest.approx(3.0)


def test_jvp_runtime_for_and_while_loops():
  def dyn_for(x, n):
    h = x
    for i in range(n):
      h = h + 0.1 * np.tanh(h * 1.1 + 0.05 * i)
    return h

  def dyn_while(x):
    h = x
    k = 0
    while np.sum(h * h) > 0.25 and k < 20:
      h = 0.7 * h + 0.05 * np.sin(h)
      k = k + 1
    return np.sum(h * h)

  rng = np.random.default_rng(3)
  x = rng.normal(size=(6,))
  dx = rng.normal(size=(6,))

  out_f, dout_f = par.jvp(dyn_for, (x, 12), (dx, 0))
  np.testing.assert_allclose(out_f, dyn_for(x, 12), rtol=1e-12)
  np.testing.assert_allclose(dout_f, _fd_jvp(lambda a: dyn_for(a, 12), (x,), (dx,)), rtol=1e-5, atol=1e-6)

  out_w, dout_w = par.jvp(dyn_while, (x,), (dx,))
  np.testing.assert_allclose(out_w, dyn_while(x), rtol=1e-12)
  np.testing.assert_allclose(dout_w, _fd_jvp(dyn_while, (x,), (dx,)), rtol=1e-5, atol=1e-6)


def test_jvp_control_flow_if_else_and_acquired_loop_tangent():
  def branchy(x, flag):
    if flag > 0.0:
      y = x * x + 1.0
    else:
      y = np.sin(x)
    return y

  def loop_acquire(x):
    acc = 0.0
    for i in range(x.shape[0]):
      acc = acc + x[i] * x[i]
    return acc

  x = np.array([1.0, 2.0, -1.5, 0.5])
  dx = np.array([0.3, -0.4, 0.2, 0.7])
  _, d1 = par.jvp(branchy, (x, 1.0), (dx, 0.0))
  np.testing.assert_allclose(d1, 2.0 * x * dx, rtol=1e-12)
  _, d2 = par.jvp(branchy, (x, -1.0), (dx, 0.0))
  np.testing.assert_allclose(d2, np.cos(x) * dx, rtol=1e-12)

  out_a, dout_a = par.jvp(loop_acquire, (x,), (dx,))
  assert out_a == pytest.approx(float(np.sum(x * x)))
  assert dout_a == pytest.approx(float(np.sum(2.0 * x * dx)))


def test_jvp_scan_fold_map_and_records_ragged():
  def fold_fn(x):
    return par.reduce(lambda a, b: 0.8 * a + np.sin(b), x, init=0.5)

  def scan_fn(x):
    return par.scan(lambda a, b: 0.7 * a + b * b, x, init=0.1)

  def map_fn(x, w):
    return par.map(lambda row: np.tanh(row @ w), x)

  rng = np.random.default_rng(4)
  x = rng.normal(size=(8,))
  dx = rng.normal(size=(8,))

  of, dof = par.jvp(fold_fn, (x,), (dx,))
  np.testing.assert_allclose(dof, _fd_jvp(fold_fn, (x,), (dx,)), rtol=1e-5, atol=1e-6)

  os_, dos = par.jvp(scan_fn, (x,), (dx,))
  np.testing.assert_allclose(dos, _fd_jvp(scan_fn, (x,), (dx,)), rtol=1e-5, atol=1e-6)

  X = rng.normal(size=(6, 4))
  W = rng.normal(size=(4, 3))
  dX = rng.normal(size=(6, 4))
  dW = rng.normal(size=(4, 3))
  om, dom = par.jvp(map_fn, (X, W), (dX, dW))
  np.testing.assert_allclose(dom, _fd_jvp(map_fn, (X, W), (dX, dW)), rtol=1e-5, atol=1e-6)

  # Record + RaggedArray with discrete metadata
  def rec_fn(layer, rag):
    y = layer.scale * (rag.values * rag.values) + layer.bias[0]
    return Pair(y, layer.weight * 2.0, tag=7), RaggedArray(y, rag.offsets)

  layer = Layer(rng.normal(size=(3,)), rng.normal(size=(2,)), 1.5, label="l1")
  dlayer = Layer(rng.normal(size=(3,)), rng.normal(size=(2,)), 0.25, label="l1")
  rag = RaggedArray.from_lengths(rng.normal(size=(5,)), np.array([2, 3], dtype=np.int64))
  drag = RaggedArray.from_lengths(rng.normal(size=(5,)), np.array([2, 3], dtype=np.int64))
  (p_out, r_out), (dp_out, dr_out) = par.jvp(rec_fn, (layer, rag), (dlayer, drag))
  assert p_out.tag == 7 and dp_out.tag == 7
  expected_dy = (
    dlayer.scale * (rag.values * rag.values) + layer.scale * (2.0 * rag.values * drag.values) + dlayer.bias[0]
  )
  np.testing.assert_allclose(dp_out.w, expected_dy, rtol=1e-12)
  np.testing.assert_allclose(dp_out.b, dlayer.weight * 2.0, rtol=1e-12)
  np.testing.assert_allclose(dr_out.values, expected_dy, rtol=1e-12)
  np.testing.assert_array_equal(dr_out.offsets, rag.offsets)


# ---------------------------------------------------------------------------
# 2. @par.custom_jvp & stop_gradient under JVP
# ---------------------------------------------------------------------------


def test_custom_jvp_and_stop_gradient():
  @par.custom_jvp
  def smooth_relu(x):
    return np.maximum(x, 0.0)

  @smooth_relu.defjvp
  def _smooth_relu_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    y = np.maximum(x, 0.0)
    # Custom straight-through sigmoid slope
    slope = 1.0 / (1.0 + np.exp(-x))
    return y, dx * slope

  x = np.array([-2.0, 0.0, 2.0])
  dx = np.array([1.0, 1.0, 1.0])
  y, dy = par.jvp(lambda a: smooth_relu(a) + par.stop_gradient(a * a), (x,), (dx,))
  np.testing.assert_allclose(y, np.maximum(x, 0.0) + x * x, rtol=1e-12)
  np.testing.assert_allclose(dy, 1.0 / (1.0 + np.exp(-x)), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. General non-scalar VJP, linearize, linear_transpose & transpose identity
# ---------------------------------------------------------------------------


def test_vjp_multi_output_and_transpose_identity():
  def f(x, w):
    z = np.tanh(x @ w)
    return z, np.sum(z * z, axis=1)

  rng = np.random.default_rng(7)
  x = rng.normal(size=(5, 4))
  w = rng.normal(size=(4, 3))
  vx = rng.normal(size=(5, 4))
  vw = rng.normal(size=(4, 3))
  u1 = rng.normal(size=(5, 3))
  u2 = rng.normal(size=(5,))

  (y1, y2), (dy1, dy2) = par.jvp(f, (x, w), (vx, vw))
  (vy1, vy2), pullback = par.vjp(f, x, w)
  gx, gw = pullback((u1, u2))

  np.testing.assert_allclose(vy1, y1, rtol=1e-12)
  np.testing.assert_allclose(vy2, y2, rtol=1e-12)
  # Transpose identity: <JVP(x; v), u> == <v, VJP(x; u)>
  lhs = float(np.sum(dy1 * u1) + np.sum(dy2 * u2))
  rhs = float(np.sum(vx * gx) + np.sum(vw * gw))
  assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


def test_linearize_and_linear_transpose():
  rng = np.random.default_rng(8)
  A = rng.normal(size=(4, 5))
  idx = np.array([0, 2, 1, 2, 0, 3], dtype=np.int64)

  def lin(v):
    return (A @ v)[idx]

  v = rng.normal(size=(5,))
  u = rng.normal(size=(6,))
  _, lin_fn = par.linearize(lin, v)
  lv = lin_fn(v)
  lt_fn = par.linear_transpose(lin, v)
  (ltu,) = lt_fn(u)
  assert float(np.sum(lv * u)) == pytest.approx(float(np.sum(v * ltu)), rel=1e-10, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. HVP (forward-over-reverse), jacfwd, jacrev, hessian, grad(grad(f))
# ---------------------------------------------------------------------------


def test_hvp_and_hessian_mlp_and_loops():
  def mlp_loss(x):
    h = np.tanh(x * 1.3 + 0.2)
    return np.sum(h * h * h + 0.5 * x * x)

  rng = np.random.default_rng(11)
  x = rng.normal(size=(7,))
  v = rng.normal(size=(7,))
  hv = par.hvp(mlp_loss, x, v)
  np.testing.assert_allclose(hv, _fd_hvp(mlp_loss, x, v), rtol=1e-4, atol=1e-5)

  H = par.hessian(mlp_loss)(x)
  np.testing.assert_allclose(H @ v, hv, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(H, H.T, rtol=1e-10, atol=1e-10)

  # HVP through a runtime-length checkpointed for-loop
  def loop_loss(x_):
    h = x_
    for _ in range(x_.shape[0]):
      h = h + 0.2 * np.tanh(h)
    return np.sum(h * h)

  hv_loop = par.hvp(loop_loss, x, v)
  np.testing.assert_allclose(hv_loop, _fd_hvp(loop_loss, x, v), rtol=1e-4, atol=1e-5)


def test_jacfwd_matches_jacrev():
  def f(x):
    return np.stack([np.sin(x[0] * x[1]), np.exp(x[1] - x[2]), x[0] + x[2] * x[2]])

  x = np.array([0.7, -0.4, 1.2])
  jf = par.jacfwd(f)(x)
  jr = par.jacrev(f)(x)
  np.testing.assert_allclose(jf, jr, rtol=1e-10, atol=1e-10)


def test_second_order_grad_lifted_cliffs():
  # pack (np.stack), cumsum, gather, scatter_add, map
  def f_stack(x):
    s = np.stack([x * x, x * x * x])
    return np.sum(s)

  def f_cumsum(x):
    c = np.cumsum(x * x)
    return np.sum(c * c)

  def f_gather(x, idx):
    g = x[idx]
    return np.sum(g * g * g)

  def f_map(x):
    m = par.map(lambda a: a * a * a, x)
    return np.sum(m)

  x = np.array([1.0, -2.0, 0.5, 1.5])
  idx = np.array([0, 2, 2, 3, 1], dtype=np.int64)

  # Verify grad(lambda a: sum(grad(f)(a))) matches finite differences of grad(f)
  for fn, args in [
    (f_stack, (x,)),
    (f_cumsum, (x,)),
    (f_gather, (x, idx)),
    (f_map, (x,)),
  ]:
    g2 = par.grad(lambda *a: np.sum(par.grad(fn, 0)(*a)), 0)(*args)
    v = np.ones_like(x)
    eps = 1e-5
    g_plus = par.grad(fn, 0)(x + eps * v, *args[1:])
    g_minus = par.grad(fn, 0)(x - eps * v, *args[1:])
    fd = (g_plus - g_minus) / (2 * eps)
    np.testing.assert_allclose(g2, fd, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Adjoints & JVP for topk / topk_fold, sort, FrozenHashTable, SortedIndex
# ---------------------------------------------------------------------------


def test_topk_and_sort_grad_and_jvp():
  def loss_topk(scores):
    vals, _ = par.topk(scores, 3, axis=-1, largest=True)
    return np.sum(vals * np.array([3.0, 2.0, 1.0]))

  scores = np.array(
    [
      [1.0, 5.0, 2.0, 4.0, 3.0],
      [-1.0, 0.5, 9.0, 2.0, 7.0],
    ]
  )
  g = par.grad(loss_topk)(scores)
  expected = np.array(
    [
      [0.0, 3.0, 0.0, 2.0, 1.0],
      [0.0, 0.0, 3.0, 1.0, 2.0],
    ]
  )
  np.testing.assert_allclose(g, expected, rtol=1e-12)

  dscores = np.arange(10, dtype=np.float64).reshape(2, 5) * 0.1
  _, dout = par.jvp(loss_topk, (scores,), (dscores,))
  assert dout == pytest.approx(float(np.sum(expected * dscores)), rel=1e-12)

  # np.sort under JVP
  x = np.array([3.0, 1.0, 4.0, 2.0])
  dx = np.array([10.0, 20.0, 30.0, 40.0])
  y_s, dy_s = par.jvp(lambda a: np.sort(a), (x,), (dx,))
  np.testing.assert_allclose(y_s, np.array([1.0, 2.0, 3.0, 4.0]))
  np.testing.assert_allclose(dy_s, np.array([20.0, 40.0, 10.0, 30.0]))


def test_frozen_hashtable_and_sorted_index_grad_and_jvp():
  keys = np.array([10, 20, 30, 40], dtype=np.int64)
  vals = np.array([1.5, -2.0, 3.5, 0.5], dtype=np.float64)
  ht = FrozenHashTable.from_items(keys, vals)
  si = SortedIndex.from_unsorted(keys, vals)
  queries = np.array([30, 10, 99, 30], dtype=np.int64)

  def ht_loss(table, qs):
    acc = 0.0
    for i in range(qs.shape[0]):
      v = table.get(qs[i], 0.0)
      acc = acc + v * v
    return acc

  def si_loss(index, qs):
    acc = 0.0
    for i in range(qs.shape[0]):
      v = index.get(qs[i], 0.0)
      acc = acc + v * v
    return acc

  # Reverse-mode grad w.r.t. FrozenHashTable (scatters cotangents into matched value slots; zero on misses)
  g_ht = par.grad(ht_loss, 0)(ht, queries)
  # Query 10 hit once -> 2*1.5=3.0; query 30 hit twice -> 2*(2*3.5)=14.0; misses (99) -> 0.0
  slot_10 = int(np.where(ht.keys == 10)[0][0])
  slot_30 = int(np.where(ht.keys == 30)[0][0])
  expected_ht_grad = np.zeros_like(ht.values)
  expected_ht_grad[slot_10] = 3.0
  expected_ht_grad[slot_30] = 14.0
  np.testing.assert_allclose(g_ht.values, expected_ht_grad, rtol=1e-12)
  np.testing.assert_array_equal(g_ht.keys, ht.keys)

  # Forward-mode JVP w.r.t. FrozenHashTable
  d_vals = np.ones_like(ht.values) * 0.5
  d_ht = FrozenHashTable(ht.keys, d_vals, ht.empty_key)
  _, dh = par.jvp(ht_loss, (ht, queries), (d_ht, queries))
  assert dh == pytest.approx(float(np.sum(expected_ht_grad * d_vals)), rel=1e-12)

  # Reverse-mode grad w.r.t. SortedIndex
  g_si = par.grad(si_loss, 0)(si, queries)
  np.testing.assert_allclose(g_si.values, np.array([3.0, 0.0, 14.0, 0.0]), rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. Idiomatic scatter lowering: np.add.at & canonical for-loop lifting
# ---------------------------------------------------------------------------


def test_np_add_at_and_for_loop_scatter_lowering():
  def scatter_add_at(idx, vals, n):
    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, idx, vals)
    return out

  def scatter_for_loop(idx, vals, n):
    out = np.zeros(n, dtype=np.float64)
    for i in range(idx.shape[0]):
      out[idx[i]] += vals[i]
    return out

  idx = np.array([1, 3, 1, 0, 3, 3], dtype=np.int64)
  vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
  expected = np.array([4.0, 4.0, 0.0, 13.0], dtype=np.float64)

  for fn in (scatter_add_at, scatter_for_loop):
    res = par.jit(fn)(idx, vals, 4)
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    # Verify both lower to IndexScatter!
    rep = par.show_rewrite(fn, idx, vals, 4)
    assert rep.after_scatters >= 1
    # Verify grad & jvp work through both!
    w = np.array([10.0, 20.0, 30.0, 40.0])
    g = par.grad(lambda v_: np.sum(fn(idx, v_, 4) * w))(vals)
    np.testing.assert_allclose(g, w[idx], rtol=1e-12)
    dv = np.arange(6, dtype=np.float64) + 1.0
    _, dout = par.jvp(lambda v_: fn(idx, v_, 4), (vals,), (dv,))
    np.testing.assert_allclose(dout, np.bincount(idx, weights=dv, minlength=4), rtol=1e-12)

  # Multi-histogram loop with shared index binding lifts AND fuses 2 -> 1 IndexScatter!
  def dual_hist_loop(idx_, g_, h_, n_):
    gh = np.zeros(n_, dtype=np.float64)
    hh = np.zeros(n_, dtype=np.float64)
    for i in range(len(idx_)):
      b = idx_[i]
      gh[b] += g_[i]
      hh[b] += h_[i]
    return gh, hh

  hvals = vals * 0.5 + 1.0
  gh_out, hh_out = par.jit(dual_hist_loop)(idx, vals, hvals, 4)
  np.testing.assert_allclose(gh_out, np.bincount(idx, weights=vals, minlength=4), rtol=1e-12)
  np.testing.assert_allclose(hh_out, np.bincount(idx, weights=hvals, minlength=4), rtol=1e-12)
  rep_dual = par.show_rewrite(dual_hist_loop, idx, vals, hvals, 4)
  assert rep_dual.before_scatters == 2 and rep_dual.after_scatters == 1


# ---------------------------------------------------------------------------
# 7. Zero-O(T^2) FlashAttention JVP & HVP, LayerNorm HVP, multi-layer loop AD
# ---------------------------------------------------------------------------


def _has_quadratic_alloc(ir_text: str, m_prefix: str = "q_n0_", n_prefix: str = "k_n0_") -> bool:
  import re

  m_match = re.search(rf"\b({re.escape(m_prefix)}\d+)\b", ir_text)
  n_match = re.search(rf"\b({re.escape(n_prefix)}\d+)\b", ir_text)
  assert m_match is not None and n_match is not None, f"missing shape params {m_prefix}*/{n_prefix}* in IR"
  m_name, n_name = m_match.group(1), n_match.group(1)
  quad_counts: set[str] = set()
  for line in ir_text.splitlines():
    s = line.strip()
    if f"({m_name} * {n_name})" in s or f"({n_name} * {m_name})" in s:
      lhs = s.split(":", 1)[0].strip()
      quad_counts.add(lhs)
  for line in ir_text.splitlines():
    if "alloc(" in line:
      for qc in quad_counts:
        if f", {qc})" in line or f", {qc}," in line:
          return True
  return False


def test_flash_attention_jvp_and_hvp_zero_quadratic_allocs():
  def flash_attn(q, k, v):
    scale = np.float64(1.0 / np.sqrt(np.float64(q.shape[-1])))
    s = (q @ k.T) * scale
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)) @ v

  def flash_loss(q, k, v):
    o = flash_attn(q, k, v)
    return np.sum(o * o)

  def run_jvp(q, k, v, dq, dk, dv):
    return par.jvp(flash_attn, (q, k, v), (dq, dk, dv))

  def run_hvp(q, k, v, dq, dk, dv):
    return par.hvp(flash_loss, (q, k, v), (dq, dk, dv))

  rng = np.random.default_rng(31)
  q, k, v = [rng.normal(size=(16, 8)) for _ in range(3)]
  dq, dk, dv = [rng.normal(size=(16, 8)) for _ in range(3)]
  eps = 1e-6

  o, do = run_jvp(q, k, v, dq, dk, dv)
  np.testing.assert_allclose(o, flash_attn(q, k, v), rtol=1e-10, atol=1e-10)
  fd_jvp = (
    flash_attn(q + eps * dq, k + eps * dk, v + eps * dv) - flash_attn(q - eps * dq, k - eps * dk, v - eps * dv)
  ) / (2 * eps)
  np.testing.assert_allclose(do, fd_jvp, rtol=1e-5, atol=1e-6)

  rep_jvp = par.show_rewrite(run_jvp, q, k, v, dq, dk, dv)
  assert any("Attention JVP" in law for law in rep_jvp.laws_triggered)
  assert not _has_quadratic_alloc(rep_jvp.after_ir)

  hq, hk, hv = run_hvp(q, k, v, dq, dk, dv)
  g = par.grad(flash_loss, (0, 1, 2))
  gp = g(q + eps * dq, k + eps * dk, v + eps * dv)
  gm = g(q - eps * dq, k - eps * dk, v - eps * dv)
  for got, p_, m_ in zip((hq, hk, hv), gp, gm):
    np.testing.assert_allclose(got, (p_ - m_) / (2 * eps), rtol=1e-5, atol=1e-6)

  rep_hvp = par.show_rewrite(run_hvp, q, k, v, dq, dk, dv)
  assert any("Attention JVP" in law for law in rep_hvp.laws_triggered)
  assert any("Attention HVP" in law for law in rep_hvp.laws_triggered)
  assert not _has_quadratic_alloc(rep_hvp.after_ir)


def test_layernorm_jvp_hvp_and_multilayer_loop_attention_ad():
  def layernorm_loss(x):
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + 1e-5)
    return np.sum(y * y * y)

  rng = np.random.default_rng(37)
  x = rng.normal(size=(8, 12))
  vx = rng.normal(size=(8, 12))
  eps = 1e-6

  _, dln = par.jvp(layernorm_loss, (x,), (vx,))
  fd_j = (layernorm_loss(x + eps * vx) - layernorm_loss(x - eps * vx)) / (2 * eps)
  assert dln == pytest.approx(fd_j, rel=1e-5, abs=1e-6)

  hln = par.hvp(layernorm_loss, x, vx)
  gln = par.grad(layernorm_loss)
  fd_h = (gln(x + eps * vx) - gln(x - eps * vx)) / (2 * eps)
  np.testing.assert_allclose(hln, fd_h, rtol=1e-5, atol=1e-6)

  # Multi-layer Transformer attention stack written as a runtime for-loop
  def multi_layer_attn(h0, wq, wk, wv):
    h = h0
    for t in range(wq.shape[0]):
      q = h @ wq[t]
      k = h @ wk[t]
      v = h @ wv[t]
      scale = np.float64(1.0 / np.sqrt(np.float64(q.shape[-1])))
      s = (q @ k.T) * scale
      e = np.exp(s - np.max(s, axis=-1, keepdims=True))
      h = h + 0.1 * ((e / np.sum(e, axis=-1, keepdims=True)) @ v)
    return np.sum(h * h)

  h0 = rng.normal(size=(8, 4))
  wq = rng.normal(size=(3, 4, 4)) * 0.2
  wk = rng.normal(size=(3, 4, 4)) * 0.2
  wv = rng.normal(size=(3, 4, 4)) * 0.2
  gh0 = par.grad(multi_layer_attn, 0)(h0, wq, wk, wv)
  fd_h0 = np.zeros_like(h0)
  for i in range(h0.shape[0]):
    for j in range(h0.shape[1]):
      e = np.zeros_like(h0)
      e[i, j] = eps
      fd_h0[i, j] = (multi_layer_attn(h0 + e, wq, wk, wv) - multi_layer_attn(h0 - e, wq, wk, wv)) / (2 * eps)
  np.testing.assert_allclose(gh0, fd_h0, rtol=1e-4, atol=1e-5)

  rep = par.show_rewrite(par.grad(multi_layer_attn, 0), h0, wq, wk, wv)
  assert any("Checkpointed Reverse Loop" in law for law in rep.laws_triggered)


def test_algebraic_ad_rules_and_multicore_verify():
  from parakeet.transforms.algebraic_vjp import ALGEBRAIC_AD_RULES

  names = {r.name for r in ALGEBRAIC_AD_RULES}
  assert {"flash_attention", "cut_cross_entropy", "softmax", "layernorm_rmsnorm"} <= names

  def f(x):
    return np.sum(np.tanh(x * 1.5) ** 3)

  x = np.linspace(-1.0, 1.0, 16)
  v = np.linspace(0.5, -0.5, 16)
  with par.options(backend="c"):
    hv_c = par.hvp(f, x, v)
  with par.options(backend="multicore"):
    hv_mc = par.hvp(f, x, v)
  np.testing.assert_allclose(hv_c, hv_mc, rtol=1e-12, atol=1e-12)
