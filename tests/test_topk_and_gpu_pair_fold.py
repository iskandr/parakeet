from typing import NamedTuple

import numpy as np
import pytest

import parakeet as pk
from parakeet import dsl
from parakeet.syntax import ir
from parakeet.transforms.pairwise_fold import COLUMNS, ROWS


def _ref_topk(x: np.ndarray, k: int, *, largest: bool = True) -> tuple[np.ndarray, np.ndarray]:
  """Exact stable NumPy reference matching dsl.topk_fold tie & NaN semantics."""
  if x.ndim == 1:
    m = x.shape[0]
    if k == 0:
      return np.empty((0,), dtype=x.dtype), np.empty((0,), dtype=np.int64)
    key = -x if (largest and np.issubdtype(x.dtype, np.floating)) else (-x.astype(np.int64) if largest else x)
    order = np.argsort(key, kind="stable")
    take = min(k, m)
    sent = (
      np.array(-np.inf if largest else np.inf, dtype=x.dtype)
      if np.issubdtype(x.dtype, np.floating)
      else np.array(np.iinfo(x.dtype).min if largest else np.iinfo(x.dtype).max, dtype=x.dtype)
    )
    vals = np.full((k,), sent, dtype=x.dtype)
    idxs = np.full((k,), -1, dtype=np.int64)
    if take > 0:
      vals[:take] = x[order[:take]]
      idxs[:take] = order[:take]
    return vals, idxs
  rows_v, rows_i = [], []
  for r in range(x.shape[0]):
    rv, ri = _ref_topk(x[r], k, largest=largest)
    rows_v.append(rv)
    rows_i.append(ri)
  return np.stack(rows_v, axis=0), np.stack(rows_i, axis=0)


def _assert_topk_equal(got_v, got_i, exp_v, exp_i):
  np.testing.assert_array_equal(got_i, exp_i)
  np.testing.assert_array_equal(np.isnan(got_v), np.isnan(exp_v))
  mask = ~np.isnan(exp_v)
  np.testing.assert_allclose(got_v[mask], exp_v[mask], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 1. Exact Top-K Monoid (1D, 2D, ties across tiles, NaNs, k > M, M == 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["c", "multicore"])
@pytest.mark.parametrize("k", [1, 2, 5, 16])
@pytest.mark.parametrize("largest", [True, False])
def test_topk_exact_ties_nans_and_edges(backend, k, largest):
  rng = np.random.default_rng(100 + k + (17 if largest else 0))

  @pk.jit
  def run_topk(arr):
    return dsl.topk_fold(k, arr, axis=-1, largest=largest)

  # 1D with exact ties across 3+ tiles and NaNs
  x1 = rng.normal(size=(3 * COLUMNS + 19,)).astype(np.float32)
  x1[[3, COLUMNS + 2, 2 * COLUMNS + 5]] = 9.5 if largest else -9.5
  x1[[7, COLUMNS + 11]] = np.nan
  for arr in (x1, x1[:3], x1[:0]):
    gv, gi = run_topk(arr, _backend=backend)
    ev, ei = _ref_topk(arr, k, largest=largest)
    _assert_topk_equal(gv, gi, ev, ei)

  # 2D with ties across tile boundaries & NaNs & k > M
  x2 = rng.normal(size=(11, 2 * COLUMNS + 7)).astype(np.float32)
  x2[0, [1, COLUMNS + 1, 2 * COLUMNS + 1]] = 12.0 if largest else -12.0
  x2[1, [4, COLUMNS + 4]] = np.nan
  for mat in (x2, x2[:, :3], x2[:, :0]):
    gv, gi = run_topk(mat, _backend=backend)
    ev, ei = _ref_topk(mat, k, largest=largest)
    _assert_topk_equal(gv, gi, ev, ei)


# ---------------------------------------------------------------------------
# 2. Pairwise k-NN & Morgan Fingerprint Tanimoto Top-k (Zero O(N*M) Alloc)
# ---------------------------------------------------------------------------


@pk.jit
def pairwise_knn5(x, y):
  d = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
  return dsl.topk_fold(5, d, axis=1, largest=False)


@pk.jit
def pairwise_knn16(x, y):
  d = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
  return dsl.topk_fold(16, d, axis=1, largest=False)


@pk.jit
def screen_tanimoto_k5(q, r):
  return dsl.tanimoto_topk(q, r, 5)


def _ref_tanimoto(q: np.ndarray, r: np.ndarray, k: int):
  # Exact uint64 popcount via byte lookup table
  lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

  def pop64(a):
    return lut[a.view(np.uint8)].reshape(*a.shape, 8).sum(axis=-1)

  pq = pop64(q).sum(axis=-1)[:, None]
  pr = pop64(r).sum(axis=-1)[None, :]
  inter = pop64(q[:, None, :] & r[None, :, :]).sum(axis=-1)
  union = pq + pr - inter
  sim = np.where(union > 0, inter.astype(np.float32) / np.maximum(union, 1).astype(np.float32), np.float32(0.0))
  return _ref_topk(sim, k, largest=True)


def test_pairwise_knn_and_tanimoto_zero_nm_alloc():
  x = np.random.default_rng(1).normal(size=(19, 8)).astype(np.float32)
  y = np.random.default_rng(2).normal(size=(2 * COLUMNS + 9, 8)).astype(np.float32)
  y[3] = y[COLUMNS + 7] = x[0]

  for fn, k in ((pairwise_knn5, 5), (pairwise_knn16, 16)):
    rep = dsl.show_rewrite(fn, x, y, backend="c")
    assert "Pairwise map and ordered tile-summary fold" in rep.laws_triggered

    lowered = fn.trace_passes(x, y, _backend="c")
    roots = {res.buf.uid for res in lowered.results if isinstance(res, ir.ArrayResult)}
    scratch = [s.count for s in ir.walk_stmts(lowered.body) if isinstance(s, ir.Alloc) and s.var.uid not in roots]
    assert scratch and all(isinstance(n, ir.Const) and 0 < n.value <= ROWS * COLUMNS for n in scratch)

    dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    ev, ei = _ref_topk(dist, k, largest=False)
    for backend in ("c", "multicore"):
      gv, gi = fn(x, y, _backend=backend)
      _assert_topk_equal(gv, gi, ev, ei)

  # Morgan 2048-bit (32 x uint64) Tanimoto screening
  rng = np.random.default_rng(7)
  q_fp = rng.integers(0, 2**63, size=(12, 32), dtype=np.uint64)
  r_fp = rng.integers(0, 2**63, size=(2 * COLUMNS + 5, 32), dtype=np.uint64)
  q_fp[0] = 0  # all-zero fingerprint edge case
  r_fp[3] = r_fp[COLUMNS + 5] = q_fp[1]  # exact 1.0 tie across tile boundary

  rep_t = dsl.show_rewrite(screen_tanimoto_k5, q_fp, r_fp, backend="c")
  assert "Pairwise map and ordered tile-summary fold" in rep_t.laws_triggered
  ev, ei = _ref_tanimoto(q_fp, r_fp, 5)
  for backend in ("c", "multicore"):
    gv, gi = screen_tanimoto_k5(q_fp, r_fp, _backend=backend)
    _assert_topk_equal(gv, gi, ev, ei)


# ---------------------------------------------------------------------------
# 3. Seeded pairwise dsl.fold (Welford mean/var + Kabsch 3x3 vector Carry)
# ---------------------------------------------------------------------------


class Welford(NamedTuple):
  n: np.float32
  mean: np.float32
  m2: np.float32


def _welford_lift(v):
  return Welford(np.float32(1.0), v, np.float32(0.0))


def _welford_merge(a, b):
  if a.n == np.float32(0.0):
    return b
  if b.n == np.float32(0.0):
    return a
  n = a.n + b.n
  delta = b.mean - a.mean
  mean = a.mean + delta * (b.n / n)
  m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / n)
  return Welford(n, mean, m2)


@pk.jit
def pairwise_welford_dist(x, y):
  d = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
  seed = Welford(np.float32(0.0), np.float32(0.0), np.float32(0.0))
  return dsl.fold(
    _welford_lift,
    _welford_merge,
    d,
    init=seed,
    axis=1,
    emit=lambda s: (s.mean, s.m2 / np.maximum(s.n, np.float32(1.0))),
  )


class Cov3x3(NamedTuple):
  n: np.float32
  mat: np.ndarray


@pk.jit
def kabsch_cov_fold(p, q):
  def lift(pair_row):
    # pair_row has shape (6,): [px, py, pz, qx, qy, qz]
    a = pair_row[:3]
    b = pair_row[3:]
    m = a[:, None] * b[None, :]
    return Cov3x3(np.float32(1.0), m)

  def merge(l, r):
    return Cov3x3(l.n + r.n, l.mat + r.mat)

  packed = np.concatenate((p, q), axis=1)
  seed = Cov3x3(np.float32(0.0), np.zeros((3, 3), dtype=np.float32))
  res = dsl.fold(lift, merge, packed, init=seed)
  return res.mat


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_seeded_pairwise_welford_and_kabsch_vector_carry(backend):
  rng = np.random.default_rng(19)
  x = rng.normal(size=(15, 6)).astype(np.float32)
  y = rng.normal(size=(2 * COLUMNS + 11, 6)).astype(np.float32)

  rep = dsl.show_rewrite(pairwise_welford_dist, x, y, backend="c")
  assert "Pairwise map and ordered tile-summary fold" in rep.laws_triggered

  mean_got, var_got = pairwise_welford_dist(x, y, _backend=backend)
  d_ref = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
  np.testing.assert_allclose(mean_got, d_ref.mean(axis=1), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(var_got, d_ref.var(axis=1), rtol=1e-4, atol=1e-4)

  # Empty reference edge case handled cleanly by seed!
  m0, v0 = pairwise_welford_dist(x, y[:0], _backend=backend)
  np.testing.assert_array_equal(m0, np.zeros(len(x), dtype=np.float32))
  np.testing.assert_array_equal(v0, np.zeros(len(x), dtype=np.float32))

  # Kabsch 3x3 covariance using fixed-size (3, 3) ArrayVal Carry
  p = rng.normal(size=(257, 3)).astype(np.float32)
  q = rng.normal(size=(257, 3)).astype(np.float32)
  cov_got = kabsch_cov_fold(p, q, _backend=backend)
  np.testing.assert_allclose(cov_got, p.T @ q, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Pair-Biased Law 1 Attention & 3D IPA Attention (Forward + pk.grad)
# ---------------------------------------------------------------------------


@pk.jit
def pair_biased_attn_2d(q, k, v, bias):
  sc = np.float32(1.0 / np.sqrt(q.shape[-1]))
  s = (q @ k.T) * sc + bias
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


@pk.jit
def pair_biased_attn_4d(q, k, v, bias):
  sc = np.float32(1.0 / np.sqrt(q.shape[-1]))
  s = (q @ np.swapaxes(k, -1, -2)) * sc + bias
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


@pk.jit
def numpy_ipa_attn_2d(q, k, v, xq, xk):
  sc = np.float32(1.0 / np.sqrt(q.shape[-1]))
  sq = np.sum((xq[:, None, :] - xk[None, :, :]) ** 2, axis=-1)
  s = (q @ k.T) * sc - np.float32(0.25) * sq
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


@pk.jit
def dsl_ipa_attn_fn(q, k, v, xq, xk, bias):
  sc = np.float32(1.0 / np.sqrt(q.shape[-1]))
  return dsl.ipa_attention(q, k, v, xq, xk, np.float32(0.25), scale=sc, bias=bias)


def _ref_attn(q, k, v, bias=None, xq=None, xk=None, gamma=0.25):
  q64, k64, v64 = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
  sc = 1.0 / np.sqrt(q.shape[-1])
  s = np.matmul(q64, np.swapaxes(k64, -1, -2)) * sc
  if xq is not None and xk is not None:
    xq64, xk64 = xq.astype(np.float64), xk.astype(np.float64)
    sq = np.sum((xq64[..., :, None, :] - xk64[..., None, :, :]) ** 2, axis=-1)
    s = s - gamma * sq
  if bias is not None:
    s = s + bias.astype(np.float64)
  s = s - np.max(s, axis=-1, keepdims=True)
  e = np.exp(s)
  return (np.matmul(e / np.sum(e, axis=-1, keepdims=True), v64)).astype(q.dtype)


def _finite_diff(loss_fn, args, arg_idx, eps=1e-3):
  x = args[arg_idx]
  g = np.zeros_like(x, dtype=np.float64)
  flat = x.reshape(-1)
  g_flat = g.reshape(-1)
  for i in range(flat.size):
    xp = x.copy()
    xm = x.copy()
    xp.reshape(-1)[i] += eps
    xm.reshape(-1)[i] -= eps
    ap = list(args)
    am = list(args)
    ap[arg_idx] = xp
    am[arg_idx] = xm
    g_flat[i] = (loss_fn(*ap) - loss_fn(*am)) / (2.0 * eps)
  return g.astype(x.dtype)


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_pair_biased_and_ipa_attention_cpu(backend):
  rng = np.random.default_rng(31)
  m, n, d = 19, 23, 8
  q = rng.normal(size=(m, d)).astype(np.float32) * 0.3
  k = rng.normal(size=(n, d)).astype(np.float32) * 0.3
  v = rng.normal(size=(n, d)).astype(np.float32) * 0.3
  bias = rng.normal(size=(m, n)).astype(np.float32) * 0.2
  xq = rng.normal(size=(m, 3)).astype(np.float32) * 0.5
  xk = rng.normal(size=(n, 3)).astype(np.float32) * 0.5
  w = rng.normal(size=(m, d)).astype(np.float32) * 0.3

  # 1. Ordinary NumPy pair-biased attention fuses into Law 1 flash_monoid
  rep_b = dsl.show_rewrite(pair_biased_attn_2d, q, k, v, bias, backend="c")
  assert "flash_monoid" in rep_b.monoidal_plans
  out_b = pair_biased_attn_2d(q, k, v, bias, _backend=backend)
  np.testing.assert_allclose(out_b, _ref_attn(q, k, v, bias=bias), rtol=1e-5, atol=1e-5)

  # 4D batched with broadcast (M, N) pair bias across (B, H)
  q4 = rng.normal(size=(2, 3, 13, 8)).astype(np.float32) * 0.3
  k4 = rng.normal(size=(2, 3, 17, 8)).astype(np.float32) * 0.3
  v4 = rng.normal(size=(2, 3, 17, 8)).astype(np.float32) * 0.3
  b4_bc = rng.normal(size=(13, 17)).astype(np.float32) * 0.2
  rep_b4 = dsl.show_rewrite(pair_biased_attn_4d, q4, k4, v4, b4_bc, backend="c")
  assert "flash_monoid" in rep_b4.monoidal_plans
  np.testing.assert_allclose(
    pair_biased_attn_4d(q4, k4, v4, b4_bc, _backend=backend), _ref_attn(q4, k4, v4, bias=b4_bc), rtol=1e-5, atol=1e-5
  )

  # 2. Law 2 pk.grad on ordinary NumPy pair-biased attention (checks dQ, dK, dV, dB!)
  def loss_pb(q_, k_, v_, b_):
    return np.sum(pair_biased_attn_2d(q_, k_, v_, b_) * w)

  grad_pb = pk.grad(loss_pb, argnum=(0, 1, 2, 3))
  rep_gpb = dsl.show_rewrite(grad_pb, q, k, v, bias, backend="c")
  assert "flash_monoid_bwd" in rep_gpb.monoidal_plans
  with pk.options(backend=backend):
    dq, dk, dv, db = grad_pb(q, k, v, bias)

  def ref_loss_pb(q_, k_, v_, b_):
    return float(np.sum(_ref_attn(q_, k_, v_, bias=b_).astype(np.float64) * w.astype(np.float64)))

  for idx, got in enumerate((dq, dk, dv, db)):
    exp = _finite_diff(ref_loss_pb, (q, k, v, bias), idx, eps=1e-3)
    np.testing.assert_allclose(got, exp, rtol=2e-3, atol=2e-3)

  # 3. Ordinary NumPy 3D IPA attention fuses into Law 1 flash_monoid
  rep_ni = dsl.show_rewrite(numpy_ipa_attn_2d, q, k, v, xq, xk, backend="c")
  assert "flash_monoid" in rep_ni.monoidal_plans
  out_ni = numpy_ipa_attn_2d(q, k, v, xq, xk, _backend=backend)
  np.testing.assert_allclose(out_ni, _ref_attn(q, k, v, xq=xq, xk=xk), rtol=1e-5, atol=1e-5)

  # 4. dsl.ipa_attention forward + Law 2 pk.grad through all 6 inputs (Q, K, V, xq, xk, bias)
  out_ipa = dsl_ipa_attn_fn(q, k, v, xq, xk, bias, _backend=backend)
  np.testing.assert_allclose(out_ipa, _ref_attn(q, k, v, bias=bias, xq=xq, xk=xk), rtol=1e-5, atol=1e-5)

  def loss_ipa(q_, k_, v_, xq_, xk_, b_):
    return np.sum(dsl_ipa_attn_fn(q_, k_, v_, xq_, xk_, b_) * w)

  grad_ipa = pk.grad(loss_ipa, argnum=(0, 1, 2, 3, 4, 5))
  rep_gipa = dsl.show_rewrite(grad_ipa, q, k, v, xq, xk, bias, backend="c")
  assert "flash_monoid_bwd" in rep_gipa.monoidal_plans
  with pk.options(backend=backend):
    grads_ipa = grad_ipa(q, k, v, xq, xk, bias)

  def ref_loss_ipa(q_, k_, v_, xq_, xk_, b_):
    return float(np.sum(_ref_attn(q_, k_, v_, bias=b_, xq=xq_, xk=xk_).astype(np.float64) * w.astype(np.float64)))

  for idx, got in enumerate(grads_ipa):
    exp = _finite_diff(ref_loss_ipa, (q, k, v, xq, xk, bias), idx, eps=1e-3)
    np.testing.assert_allclose(got, exp, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# 5. GPU Verification (CUDA Simulator & Native Metal with device_fallback="error")
# ---------------------------------------------------------------------------


def _check_gpu_all(backend: str):
  rng = np.random.default_rng(55)
  x = rng.normal(size=(17, 8)).astype(np.float32)
  y = rng.normal(size=(165, 8)).astype(np.float32)
  x[0] = y[4] = y[133]

  q_fp = rng.integers(0, 2**63, size=(9, 32), dtype=np.uint64)
  r_fp = rng.integers(0, 2**63, size=(145, 32), dtype=np.uint64)
  r_fp[2] = r_fp[101] = q_fp[0]

  m, n, d = 19, 23, 8
  q = rng.normal(size=(m, d)).astype(np.float32) * 0.3
  k = rng.normal(size=(n, d)).astype(np.float32) * 0.3
  v = rng.normal(size=(n, d)).astype(np.float32) * 0.3
  bias = rng.normal(size=(m, n)).astype(np.float32) * 0.2
  xq = rng.normal(size=(m, 3)).astype(np.float32) * 0.5
  xk = rng.normal(size=(n, 3)).astype(np.float32) * 0.5
  w = rng.normal(size=(m, d)).astype(np.float32) * 0.3

  with pk.options(backend=backend, device_threshold=1, device_fallback="error"):
    # Pairwise k-NN (k=5 and k=16 — tests packed dtype summary buffers staying under Metal's 28-buffer cap!)
    dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    for fn, kv in ((pairwise_knn5, 5), (pairwise_knn16, 16)):
      gv, gi = fn(x, y)
      ev, ei = _ref_topk(dist, kv, largest=False)
      _assert_topk_equal(gv, gi, ev, ei)

    # Tanimoto top-k on GPU
    tv, ti = screen_tanimoto_k5(q_fp, r_fp)
    etv, eti = _ref_tanimoto(q_fp, r_fp, 5)
    _assert_topk_equal(tv, ti, etv, eti)

    # Seeded pairwise Welford on GPU
    wm, wv = pairwise_welford_dist(x, y)
    np.testing.assert_allclose(wm, dist.mean(axis=1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(wv, dist.var(axis=1), rtol=1e-4, atol=1e-4)

    # Pair-biased attention forward + pk.grad on GPU
    np.testing.assert_allclose(pair_biased_attn_2d(q, k, v, bias), _ref_attn(q, k, v, bias=bias), rtol=1e-5, atol=1e-5)

    def loss_pb(q_, k_, v_, b_):
      return np.sum(pair_biased_attn_2d(q_, k_, v_, b_) * w)

    dq, dk, dv, db = pk.grad(loss_pb, argnum=(0, 1, 2, 3))(q, k, v, bias)

    def ref_loss_pb(q_, k_, v_, b_):
      return float(np.sum(_ref_attn(q_, k_, v_, bias=b_).astype(np.float64) * w.astype(np.float64)))

    for idx, got in enumerate((dq, dk, dv, db)):
      exp = _finite_diff(ref_loss_pb, (q, k, v, bias), idx, eps=1e-3)
      np.testing.assert_allclose(got, exp, rtol=2e-3, atol=2e-3)

    # 3D IPA attention (NumPy forward + dsl.ipa_attention forward & pk.grad) on GPU
    np.testing.assert_allclose(
      numpy_ipa_attn_2d(q, k, v, xq, xk), _ref_attn(q, k, v, xq=xq, xk=xk), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
      dsl_ipa_attn_fn(q, k, v, xq, xk, bias), _ref_attn(q, k, v, bias=bias, xq=xq, xk=xk), rtol=1e-5, atol=1e-5
    )

    def loss_ipa(q_, k_, v_, xq_, xk_, b_):
      return np.sum(dsl_ipa_attn_fn(q_, k_, v_, xq_, xk_, b_) * w)

    grads_ipa = pk.grad(loss_ipa, argnum=(0, 1, 2, 3, 4, 5))(q, k, v, xq, xk, bias)

    def ref_loss_ipa(q_, k_, v_, xq_, xk_, b_):
      return float(np.sum(_ref_attn(q_, k_, v_, bias=b_, xq=xq_, xk=xk_).astype(np.float64) * w.astype(np.float64)))

    for idx, got in enumerate(grads_ipa):
      exp = _finite_diff(ref_loss_ipa, (q, k, v, xq, xk, bias), idx, eps=1e-3)
      np.testing.assert_allclose(got, exp, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(pk.available_backends().get("metal") != "ok", reason="Metal device required")
def test_metal_topk_tanimoto_and_pair_bias():
  _check_gpu_all("metal")


def test_cuda_simulated_topk_tanimoto_and_pair_bias(tmp_path):
  import sys

  from tests.test_backends import run_script

  if sys.platform == "win32":
    pytest.skip("the CUDA simulator needs a POSIX C++ toolchain")
  proc = run_script(
    tmp_path,
    "from tests.test_topk_and_gpu_pair_fold import _check_gpu_all\n_check_gpu_all('cuda')",
    PARAKEET_CUDA_SIMULATE="1",
  )
  assert proc.returncode == 0, proc.stdout + proc.stderr
