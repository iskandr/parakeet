"""Tests for #52: Law 5 generalized — 3D Morton ordering, 3D AABB spatial tile pruning,
and CSR BlockMask block-sparse FlashAttention (forward + Law 2 backward)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import parakeet as pk
from parakeet import BlockMask, dsl


# ---------------------------------------------------------------------------
# 1. 3D Morton Z-order spatial sorting & 3D AABB tile pruning
# ---------------------------------------------------------------------------


def test_morton_order_and_aabb_cutoff_sum():
  rng = np.random.default_rng(52)
  # Clustered 3D point cloud (like two protein domains separated in space)
  c1 = rng.uniform(0.0, 10.0, size=(60, 3)).astype(np.float32)
  c2 = rng.uniform(40.0, 50.0, size=(60, 3)).astype(np.float32)
  coords = np.vstack([c1, c2])
  # Shuffle to test Morton ordering restores locality
  perm_in = rng.permutation(len(coords))
  coords = coords[perm_in]
  vals = rng.standard_normal((len(coords), 2)).astype(np.float32)

  @pk.jit
  def order_coords(pts):
    return dsl.morton_order(pts)

  order = order_coords(coords)
  assert order.shape == (len(coords),)
  np.testing.assert_array_equal(np.sort(order), np.arange(len(coords), dtype=np.int64))
  ordered_coords = coords[order]
  ordered_vals = vals[order]

  cutoff = np.float32(5.0)

  @pk.jit
  def rbf_cutoff(cq, ck, v, cut):
    return dsl.spatial_cutoff_sum(cq, ck, v, cut, lambda d2: np.exp(-np.float32(0.1) * d2), tile_size=16)

  out = rbf_cutoff(ordered_coords, ordered_coords, ordered_vals, cutoff)
  diff = ordered_coords[:, None, :].astype(np.float64) - ordered_coords[None, :, :].astype(np.float64)
  d2 = np.sum(diff * diff, axis=-1)
  w = np.where(d2 <= float(cutoff) ** 2, np.exp(-0.1 * d2), 0.0)
  ref = (w @ ordered_vals.astype(np.float64)).astype(np.float32)
  np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

  # NaN coordinate must disable AABB pruning for its tile (exact NaN semantics)
  coords_nan = ordered_coords.copy()
  coords_nan[5, 1] = np.nan
  out_nan = rbf_cutoff(coords_nan, coords_nan, ordered_vals, cutoff)
  diff_n = coords_nan[:, None, :].astype(np.float64) - coords_nan[None, :, :].astype(np.float64)
  d2_n = np.sum(diff_n * diff_n, axis=-1)
  w_n = np.where(d2_n <= float(cutoff) ** 2, np.exp(-0.1 * d2_n), 0.0)
  ref_n = (w_n @ ordered_vals.astype(np.float64)).astype(np.float32)
  np.testing.assert_allclose(out_nan, ref_n, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. CSR BlockMask block-sparse FlashAttention (forward + Law 2 backward)
# ---------------------------------------------------------------------------


def _ref_masked_attn(q, k, v, mask, scale=None):
  sc = (1.0 / np.sqrt(q.shape[-1])) if scale is None else float(scale)
  scores = np.matmul(q.astype(np.float64), np.swapaxes(k.astype(np.float64), -1, -2)) * sc
  scores = np.where(mask, scores, -np.inf)
  has_any = np.any(mask, axis=-1, keepdims=True)
  m = np.where(has_any, scores.max(axis=-1, keepdims=True), 0.0)
  e = np.where(mask, np.exp(scores - m), 0.0)
  denom = e.sum(axis=-1, keepdims=True)
  p = np.where(has_any, e / np.where(has_any, denom, 1.0), 0.0)
  return np.matmul(p, v.astype(np.float64)).astype(q.dtype)


def _num_grad(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
  g = np.zeros_like(x)
  fx, fg = x.ravel(), g.ravel()
  for i in range(fx.size):
    orig = fx[i]
    fx[i] = orig + eps
    fp = float(fn(x))
    fx[i] = orig - eps
    fm = float(fn(x))
    fx[i] = orig
    fg[i] = (fp - fm) / (2 * eps)
  return g


def test_block_sparse_attention_forward_backward_and_empty_rows():
  rng = np.random.default_rng(77)
  B, H, M, N, K_dim, D_dim = 1, 2, 18, 20, 6, 4
  q = rng.standard_normal((B, H, M, K_dim)).astype(np.float64) * 0.3
  k = rng.standard_normal((B, H, N, K_dim)).astype(np.float64) * 0.3
  v = rng.standard_normal((B, H, N, D_dim)).astype(np.float64) * 0.3
  w_loss = rng.standard_normal((B, H, M, D_dim)).astype(np.float64)

  # Block mask with completely empty query block (block 1 = rows 6..11) + fine elementwise mask
  dense_mask = np.zeros((M, N), dtype=bool)
  dense_mask[:6, :12] = rng.random((6, 12)) > 0.3
  dense_mask[:6, 0] = True
  # rows 6..11 intentionally ALL False -> empty row in BlockMask!
  dense_mask[12:, 8:] = rng.random((M - 12, N - 8)) > 0.3
  dense_mask[12:, 10] = True

  bm = BlockMask.from_dense_mask(dense_mask, block_q=6, block_k=6)
  assert bm.row_ptr[1] == bm.row_ptr[2]  # confirm block 1 is empty

  @pk.jit
  def fwd(q_, k_, v_, mask_b, fine_m):
    return dsl.block_sparse_attention(q_, k_, v_, mask_b, mask=fine_m)

  out = fwd(q, k, v, bm, dense_mask)
  ref = _ref_masked_attn(q, k, v, dense_mask)
  np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)
  np.testing.assert_array_equal(out[:, :, 6:12, :], 0.0)

  # Law 2 D_i-recomputed backward via pk.grad
  def loss_fn(q_, k_, v_):
    o = dsl.block_sparse_attention(q_, k_, v_, bm, mask=dense_mask)
    return np.sum(o * w_loss)

  dq, dk, dv = pk.grad(loss_fn, argnum=(0, 1, 2))(q, k, v)
  ref_dq = _num_grad(lambda x: float(np.sum(_ref_masked_attn(x, k, v, dense_mask) * w_loss)), q.copy())
  ref_dk = _num_grad(lambda x: float(np.sum(_ref_masked_attn(q, x, v, dense_mask) * w_loss)), k.copy())
  ref_dv = _num_grad(lambda x: float(np.sum(_ref_masked_attn(q, k, x, dense_mask) * w_loss)), v.copy())
  np.testing.assert_allclose(dq, ref_dq, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dk, ref_dk, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dv, ref_dv, rtol=1e-4, atol=1e-4)

  # Banded causal BlockMask + 3D cutoff BlockMask constructors
  bm_band = BlockMask.banded(M, M, band=5, block_q=6, block_k=6, causal=True)
  idx = np.arange(M)
  band_fine = (idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) <= 5)
  out_b = fwd(q, k[:, :, :M, :], v[:, :, :M, :], bm_band, band_fine)
  ref_b = _ref_masked_attn(q, k[:, :, :M, :], v[:, :, :M, :], band_fine)
  np.testing.assert_allclose(out_b, ref_b, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. GPU verification (CUDA simulator + native Metal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_spatial_and_block_sparse_on_gpu(backend):
  if backend == "cuda" and not os.environ.get("PARAKEET_CUDA_SIMULATE"):
    if pk.available_backends().get("cuda") != "ok":
      pytest.skip("CUDA not available")
  if backend == "metal" and pk.available_backends().get("metal") != "ok":
    pytest.skip("Metal not available")

  rng = np.random.default_rng(31)
  N = 128
  coords = rng.uniform(0.0, 20.0, size=(N, 3)).astype(np.float32)
  vals = rng.standard_normal(N).astype(np.float32)
  cutoff = np.float32(6.0)

  q = rng.standard_normal((2, N, 16)).astype(np.float32) * np.float32(0.25)
  k = rng.standard_normal((2, N, 16)).astype(np.float32) * np.float32(0.25)
  v = rng.standard_normal((2, N, 16)).astype(np.float32) * np.float32(0.25)
  bm = BlockMask.from_3d_cutoff(coords, coords, float(cutoff), block_q=16, block_k=16)

  @pk.jit(backend=backend)
  def gpu_spatial(pts, vs, cut):
    return dsl.spatial_cutoff_sum(pts, pts, vs, cut, lambda d2: np.float32(1.0) / (np.float32(1.0) + d2), tile_size=16)

  @pk.jit(backend=backend)
  def gpu_bsa(q_, k_, v_, mask_b):
    return dsl.block_sparse_attention(q_, k_, v_, mask_b)

  with pk.options(device_fallback="error", device_threshold=1):
    sp_out = pk.to_host(gpu_spatial(*pk.to_device((coords, vals), backend=backend), cutoff))
    bsa_out = pk.to_host(gpu_bsa(*pk.to_device((q, k, v, bm), backend=backend)))

  diff = coords[:, None, :].astype(np.float64) - coords[None, :, :].astype(np.float64)
  d2 = np.sum(diff * diff, axis=-1)
  w = np.where(d2 <= float(cutoff) ** 2, 1.0 / (1.0 + d2), 0.0)
  np.testing.assert_allclose(sp_out, (w @ vals.astype(np.float64)).astype(np.float32), rtol=1e-4, atol=1e-4)

  # Verify block_sparse_attention matches exact block mask
  dense_bm = np.zeros((N, N), dtype=bool)
  for qb in range(bm.num_q_blocks):
    for p in range(bm.row_ptr[qb], bm.row_ptr[qb + 1]):
      kb = int(bm.col_idx[p])
      dense_bm[qb * 16 : (qb + 1) * 16, kb * 16 : (kb + 1) * 16] = True
  np.testing.assert_allclose(bsa_out, _ref_masked_attn(q, k, v, dense_bm), rtol=1e-4, atol=1e-4)
