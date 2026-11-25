"""Tests for Wave 6 (v1.7.0, #82 & #83 — Irregular Trees & Adaptive 3-Tier Ragged Parallelism):
- par.partition_by_size and par.segmented_parfor 3-tier adaptive dispatch on
  skewed / power-law segment distributions (including 100k + 1,000x20), verified
  on c, multicore, and native Apple Metal (device_fallback="error")
- par.segmented_histogram2d single-pass fused (grad, hess, count) 2D scatter
  and histogram subtraction identity (parent - small == large)
- par.ragged_partition parallel CSR sample-index partitioning
- End-to-end CudaTree adaptive BFS -> DFS GBDT trainer verified against an
  independent exact CPU reference across balanced, heavily skewed, tied,
  constant-feature, and empty-split datasets
"""

from __future__ import annotations

import numpy as np

import parakeet as par
from examples.cudatree_gbdt import fit_cudatree_gbdt, predict_cudatree_gbdt


# ---------------------------------------------------------------------------
# 1. Adaptive 3-tier ragged dispatch on skewed / power-law workloads
# ---------------------------------------------------------------------------


def test_adaptive_three_tier_ragged_dispatch_skewed_and_metal():
  # Skewed power-law distribution: 1 heavy segment (100,000), 10 medium (256), 1,000 light (20)
  lengths = np.array([100_000] + [256] * 10 + [20] * 1000, dtype=np.int64)
  offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
  np.cumsum(lengths, out=offsets[1:])
  total = int(offsets[-1])
  rng = np.random.default_rng(13)
  vals = rng.standard_normal(total).astype(np.float32)

  @par.jit
  def classify_and_sum(vals, offsets):
    heavy, medium, light = par.partition_by_size(offsets, block_thresh=1024, warp_thresh=64)
    out = np.zeros(len(offsets) - 1, dtype=np.float32)

    def seg_sum(seg_id, start, stop):
      acc = np.float32(0.0)
      for k in range(start, stop):
        acc = acc + vals[k]
      out[seg_id] = acc

    par.segmented_parfor(seg_sum, offsets, block_thresh=1024, warp_thresh=64)
    return heavy, medium, light, out

  ref = np.array([vals[offsets[i] : offsets[i + 1]].sum() for i in range(len(lengths))], dtype=np.float32)

  for backend in ("c", "multicore"):
    heavy, medium, light, out = classify_and_sum(vals, offsets, _backend=backend)
    np.testing.assert_array_equal(heavy, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(medium, np.arange(1, 11, dtype=np.int64))
    np.testing.assert_array_equal(light, np.arange(11, 1011, dtype=np.int64))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-3)

  if par.available_backends().get("metal") == "ok":
    with par.options(device_fallback="error", device_threshold=1):
      heavy_m, medium_m, light_m, out_m = classify_and_sum(vals, offsets, _backend="metal")
    assert len(heavy_m) == 1 and len(medium_m) == 10 and len(light_m) == 1000
    np.testing.assert_allclose(out_m, ref, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# 2. Fused segmented 2D histogram scatter & histogram subtraction
# ---------------------------------------------------------------------------


def test_segmented_histogram2d_and_subtraction():
  rng = np.random.default_rng(29)
  n = 4096
  n_nodes = 8
  n_bins = 32
  node_ids = rng.integers(-1, n_nodes, size=n, dtype=np.int64)  # includes -1 (ignored samples)
  bins = rng.integers(0, n_bins, size=n, dtype=np.int32)
  grads = rng.standard_normal(n).astype(np.float32)
  hess = rng.uniform(0.1, 2.0, size=n).astype(np.float32)

  @par.jit
  def hist_and_subtract(node_ids, bins, grads, hess, is_left):
    pg, ph, pc = par.segmented_histogram2d(node_ids, bins, grads, hess, n_nodes, n_bins)
    left_nids = np.where(is_left, node_ids, np.int64(-1))
    right_nids = np.where(is_left, np.int64(-1), node_ids)
    lg, lh, lc = par.segmented_histogram2d(left_nids, bins, grads, hess, n_nodes, n_bins)
    rg_sub = pg - lg
    rh_sub = ph - lh
    rc_sub = pc - lc
    rg_dir, rh_dir, rc_dir = par.segmented_histogram2d(right_nids, bins, grads, hess, n_nodes, n_bins)
    return pg, ph, pc, rg_sub, rh_sub, rc_sub, rg_dir, rh_dir, rc_dir

  is_left = rng.random(n) < 0.3
  for backend in ("c", "multicore"):
    pg, ph, pc, rg_sub, rh_sub, rc_sub, rg_dir, rh_dir, rc_dir = hist_and_subtract(
      node_ids, bins, grads, hess, is_left, _backend=backend
    )
    # Verify exact agreement with numpy reference
    ref_g = np.zeros((n_nodes, n_bins), dtype=np.float32)
    ref_c = np.zeros((n_nodes, n_bins), dtype=np.int64)
    for k in range(n):
      if node_ids[k] >= 0:
        ref_g[node_ids[k], bins[k]] += grads[k]
        ref_c[node_ids[k], bins[k]] += 1
    np.testing.assert_allclose(pg, ref_g, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(pc, ref_c)
    # Histogram subtraction identity: parent - small == direct large
    np.testing.assert_allclose(rg_sub, rg_dir, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rh_sub, rh_dir, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(rc_sub, rc_dir)


# ---------------------------------------------------------------------------
# 3. Ragged sample-index partitioning
# ---------------------------------------------------------------------------


def test_ragged_partition_parallel_and_metal():
  lengths = np.array([100, 17, 0, 512, 33], dtype=np.int64)
  offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
  np.cumsum(lengths, out=offsets[1:])
  total = int(offsets[-1])
  rng = np.random.default_rng(5)
  sample_ids = rng.permutation(total).astype(np.int64)
  goes_left = rng.random(total) < 0.4

  @par.jit
  def do_part(sample_ids, offsets, goes_left):
    return par.ragged_partition(sample_ids, offsets, goes_left)

  for backend in ("c", "multicore"):
    out_ids, lc, rc = do_part(sample_ids, offsets, goes_left, _backend=backend)
    for b in range(len(lengths)):
      s, e = int(offsets[b]), int(offsets[b + 1])
      seg = sample_ids[s:e]
      mask = goes_left[s:e]
      expected = np.concatenate([seg[mask], seg[~mask]])
      np.testing.assert_array_equal(out_ids[s:e], expected)
      assert int(lc[b]) == int(np.sum(mask))
      assert int(rc[b]) == int(np.sum(~mask))

  if par.available_backends().get("metal") == "ok":
    with par.options(device_fallback="error", device_threshold=1):
      out_m, lc_m, rc_m = do_part(sample_ids, offsets, goes_left, _backend="metal")
    np.testing.assert_array_equal(out_m, out_ids)
    np.testing.assert_array_equal(lc_m, lc)
    np.testing.assert_array_equal(rc_m, rc)


# ---------------------------------------------------------------------------
# 4. End-to-end CudaTree BFS -> DFS GBDT exact agreement vs CPU reference
# ---------------------------------------------------------------------------


def _ref_gbdt_tree(X_bin, grads, hess, n_bins, max_depth, min_samples=4, reg_lambda=1.0, min_child_weight=1.0):
  """Independent pure-NumPy reference exact histogram GBDT tree builder."""
  max_nodes = (1 << (max_depth + 1)) - 1
  n_features = X_bin.shape[1]
  node_feat = np.full(max_nodes, -1, dtype=np.int64)
  node_thresh = np.full(max_nodes, -1, dtype=np.int64)
  node_left = np.full(max_nodes, -1, dtype=np.int64)
  node_right = np.full(max_nodes, -1, dtype=np.int64)
  node_value = np.zeros(max_nodes, dtype=np.float32)

  def build(u, idx, depth):
    g_tot = float(np.sum(grads[idx], dtype=np.float32))
    h_tot = float(np.sum(hess[idx], dtype=np.float32))
    node_value[u] = np.float32(-g_tot / (h_tot + reg_lambda))
    if depth >= max_depth or len(idx) < min_samples:
      return
    best_gain = 0.0
    best_feat = -1
    best_bin = -1
    for f in range(n_features):
      gh = np.zeros(n_bins, dtype=np.float32)
      hh = np.zeros(n_bins, dtype=np.float32)
      for sid in idx:
        bi = int(X_bin[sid, f])
        gh[bi] += grads[sid]
        hh[bi] += hess[sid]
      gl = np.float32(0.0)
      hl = np.float32(0.0)
      for b in range(n_bins - 1):
        gl = np.float32(gl + gh[b])
        hl = np.float32(hl + hh[b])
        hr = np.float32(h_tot - hl)
        if hl >= min_child_weight and hr >= min_child_weight:
          gr = np.float32(g_tot - gl)
          gain = float(
            (gl * gl) / (hl + reg_lambda) + (gr * gr) / (hr + reg_lambda) - (g_tot * g_tot) / (h_tot + reg_lambda)
          )
          if gain > best_gain + 1e-7:
            best_gain = gain
            best_feat = f
            best_bin = b
    if best_feat < 0:
      return
    mask = X_bin[idx, best_feat] <= best_bin
    left_idx = idx[mask]
    right_idx = idx[~mask]
    if len(left_idx) == 0 or len(right_idx) == 0:
      return
    lc, rc = 2 * u + 1, 2 * u + 2
    node_feat[u] = best_feat
    node_thresh[u] = best_bin
    node_left[u] = lc
    node_right[u] = rc
    build(lc, left_idx, depth + 1)
    build(rc, right_idx, depth + 1)

  build(0, np.arange(len(grads), dtype=np.int64), 0)
  return node_feat, node_thresh, node_left, node_right, node_value


def test_cudatree_gbdt_exact_agreement_balanced_skewed_and_edge_cases():
  rng = np.random.default_rng(101)
  n_bins = 16
  max_depth = 5

  # Case 1: Balanced dataset with BFS -> DFS handoff active (dfs_thresh=48)
  X_bal = rng.integers(0, n_bins, size=(512, 8), dtype=np.int32)
  y_bal = (
    1.2 * (X_bal[:, 0] > 7).astype(np.float32)
    - 0.9 * (X_bal[:, 3] < 5).astype(np.float32)
    + 0.4 * (X_bal[:, 5] > 11).astype(np.float32)
    + 0.1 * rng.standard_normal(512).astype(np.float32)
  )
  g_bal = -y_bal
  h_bal = np.ones(512, dtype=np.float32)

  # Case 2: Heavily skewed dataset with constant feature, ties, and tiny leaves
  X_skew = np.zeros((400, 6), dtype=np.int32)
  X_skew[:, 0] = 3  # constant feature!
  X_skew[:25, 1] = rng.integers(8, n_bins, size=25, dtype=np.int32)  # heavily skewed split (25 vs 375)
  X_skew[25:, 1] = 2
  X_skew[:, 2] = rng.integers(0, 4, size=400, dtype=np.int32)  # heavy ties
  X_skew[:, 3:] = rng.integers(0, n_bins, size=(400, 3), dtype=np.int32)
  y_skew = (
    2.0 * (X_skew[:, 1] > 5).astype(np.float32)
    + 0.7 * (X_skew[:, 2] >= 2).astype(np.float32)
    - 0.4 * (X_skew[:, 4] < 6).astype(np.float32)
  )
  g_skew = -y_skew
  h_skew = np.ones(400, dtype=np.float32)

  # Case 3: Purely constant features -> zero splits (empty split edge case)
  X_const = np.full((64, 4), 5, dtype=np.int32)
  g_const = rng.standard_normal(64).astype(np.float32)
  h_const = np.ones(64, dtype=np.float32)

  for X_bin, grads, hess in ((X_bal, g_bal, h_bal), (X_skew, g_skew, h_skew), (X_const, g_const, h_const)):
    ref_tree = _ref_gbdt_tree(X_bin, grads, hess, n_bins, max_depth)
    ref_preds = predict_cudatree_gbdt(X_bin, *ref_tree, _backend="python")
    for backend in ("c", "multicore"):
      got_tree = fit_cudatree_gbdt(X_bin, grads, hess, n_bins, max_depth, dfs_thresh=48, _backend=backend)
      np.testing.assert_array_equal(got_tree[0], ref_tree[0])
      np.testing.assert_array_equal(got_tree[1], ref_tree[1])
      np.testing.assert_array_equal(got_tree[2], ref_tree[2])
      np.testing.assert_array_equal(got_tree[3], ref_tree[3])
      np.testing.assert_allclose(got_tree[4], ref_tree[4], rtol=1e-4, atol=1e-4)
      got_preds = predict_cudatree_gbdt(X_bin, *got_tree, _backend=backend)
      np.testing.assert_allclose(got_preds, ref_preds, rtol=1e-4, atol=1e-4)
