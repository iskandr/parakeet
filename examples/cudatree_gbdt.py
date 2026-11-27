"""Histogram GBDT tree builders in Parakeet: pure BFS vs. adaptive BFS-to-DFS.

This module shows two ways to structure a histogram GBDT builder over the same
shared split-finding and partitioning primitives:

1. ``fit_gbdt_bfs``: level-wise breadth-first growth with histogram subtraction
   (build the smaller child's histogram and subtract from the parent for the
   larger child).
2. ``fit_cudatree_gbdt``: CudaTree-style adaptive BFS-to-DFS growth. Shallow
   levels run feature-parallel BFS with histogram subtraction; once a node's
   sample count falls below ``dfs_thresh``, it is deferred to a single
   thread-per-subtree DFS kernel backed by thread-local ``par.fixed_list``
   stacks.

Run::

    python examples/cudatree_gbdt.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import parakeet as par


@dataclass
class TreeConfig:
  n_features: int
  n_bins: int
  max_depth: int
  min_samples: int
  reg_lambda: np.float32
  min_child_weight: np.float32


@dataclass
class TreeNodes:
  feature: np.ndarray
  threshold: np.ndarray
  left: np.ndarray
  right: np.ndarray
  value: np.ndarray

  def as_tuple(self):
    return self.feature, self.threshold, self.left, self.right, self.value


def _alloc_tree_nodes(max_depth: int) -> TreeNodes:
  max_nodes = (1 << (max_depth + 1)) - 1
  return TreeNodes(
    feature=np.full(max_nodes, -1, dtype=np.int64),
    threshold=np.full(max_nodes, -1, dtype=np.int64),
    left=np.full(max_nodes, -1, dtype=np.int64),
    right=np.full(max_nodes, -1, dtype=np.int64),
    value=np.zeros(max_nodes, dtype=np.float32),
  )


def _split_gain(g_left, h_left, g_right, h_right, g_parent, h_parent, reg_lambda):
  return (
    (g_left * g_left) / (h_left + reg_lambda)
    + (g_right * g_right) / (h_right + reg_lambda)
    - (g_parent * g_parent) / (h_parent + reg_lambda)
  )


def _best_split_from_hist(g_hist, h_hist, cfg: TreeConfig):
  """Find the best ``(feature, bin, gain)`` across ``(F, B)`` grad/hess histograms."""
  best_gain = np.float32(0.0)
  best_feat = -1
  best_bin = -1
  g_tot = np.sum(g_hist[0])
  h_tot = np.sum(h_hist[0])
  for f in range(cfg.n_features):
    gl = np.float32(0.0)
    hl = np.float32(0.0)
    for b in range(cfg.n_bins - 1):
      gl += g_hist[f, b]
      hl += h_hist[f, b]
      hr = h_tot - hl
      if hl >= cfg.min_child_weight and hr >= cfg.min_child_weight:
        gain = _split_gain(gl, hl, g_tot - gl, hr, g_tot, h_tot, cfg.reg_lambda)
        if gain > best_gain + np.float32(1e-7):
          best_gain, best_feat, best_bin = gain, f, b
  return best_feat, best_bin, best_gain


def _partition_segment(X_bin, sample_ids, scratch_ids, lo, hi, feat, thresh):
  """Stable partition of ``sample_ids[lo:hi]`` by ``X_bin[sid, feat] <= thresh``."""
  n_left = 0
  for k in range(lo, hi):
    sid = sample_ids[k]
    if X_bin[sid, feat] <= thresh:
      scratch_ids[lo + n_left] = sid
      n_left += 1
  n_right = 0
  mid = lo + n_left
  for k in range(lo, hi):
    sid = sample_ids[k]
    if X_bin[sid, feat] > thresh:
      scratch_ids[mid + n_right] = sid
      n_right += 1
  sample_ids[lo:hi] = scratch_ids[lo:hi]
  return n_left, n_right


def _build_node_hist_parallel(X_bin, grads, hess, sample_ids, start, stop, g_out, h_out, cfg: TreeConfig):
  """Feature-parallel gradient/hessian histogram over ``sample_ids[start:stop]``."""

  def build_feature(f):
    g_out[f, : cfg.n_bins] = 0.0
    h_out[f, : cfg.n_bins] = 0.0
    for k in range(start, stop):
      sid = sample_ids[k]
      bi = X_bin[sid, f]
      g_out[f, bi] += grads[sid]
      h_out[f, bi] += hess[sid]

  par.parfor(build_feature, cfg.n_features)


def _dfs_complete_subtree(
  root_node,
  root_start,
  root_stop,
  root_depth,
  X_bin,
  grads,
  hess,
  sample_ids,
  scratch_ids,
  cfg: TreeConfig,
  nodes: TreeNodes,
):
  """Finish one small subtree on a single thread with ``par.fixed_list`` scratch.

  Complete-binary-tree indexing (``left = 2*u + 1``, ``right = 2*u + 2``) gives
  each subtree disjoint node slots so concurrent DFS threads never contend.
  """
  stk_node = par.fixed_list(64, np.int64)
  stk_lo = par.fixed_list(64, np.int64)
  stk_hi = par.fixed_list(64, np.int64)
  stk_dep = par.fixed_list(64, np.int64)
  stk_node.append(root_node)
  stk_lo.append(root_start)
  stk_hi.append(root_stop)
  stk_dep.append(root_depth)

  gh = par.fixed_list(64, np.float32)
  hh = par.fixed_list(64, np.float32)
  for _ in range(cfg.n_bins):
    gh.append(np.float32(0.0))
    hh.append(np.float32(0.0))

  while len(stk_node) > 0:
    u = stk_node.pop()
    lo = stk_lo.pop()
    hi = stk_hi.pop()
    dep = stk_dep.pop()

    g_tot = np.float32(0.0)
    h_tot = np.float32(0.0)
    for k in range(lo, hi):
      sid = sample_ids[k]
      g_tot += grads[sid]
      h_tot += hess[sid]
    nodes.value[u] = -g_tot / (h_tot + cfg.reg_lambda)

    if dep >= cfg.max_depth or (hi - lo) < cfg.min_samples:
      continue

    best_gain = np.float32(0.0)
    best_feat = -1
    best_bin = -1

    for f in range(cfg.n_features):
      for b in range(cfg.n_bins):
        gh[b] = np.float32(0.0)
        hh[b] = np.float32(0.0)
      for k in range(lo, hi):
        sid = sample_ids[k]
        bi = X_bin[sid, f]
        gh[bi] = gh[bi] + grads[sid]
        hh[bi] = hh[bi] + hess[sid]

      gl = np.float32(0.0)
      hl = np.float32(0.0)
      for b in range(cfg.n_bins - 1):
        gl += gh[b]
        hl += hh[b]
        hr = h_tot - hl
        if hl >= cfg.min_child_weight and hr >= cfg.min_child_weight:
          gain = _split_gain(gl, hl, g_tot - gl, hr, g_tot, h_tot, cfg.reg_lambda)
          if gain > best_gain + np.float32(1e-7):
            best_gain, best_feat, best_bin = gain, f, b

    if best_feat < 0:
      continue

    nl, nr = _partition_segment(X_bin, sample_ids, scratch_ids, lo, hi, best_feat, best_bin)
    if nl == 0 or nr == 0:
      continue

    mid = lo + nl
    lc, rc = 2 * u + 1, 2 * u + 2
    nodes.feature[u], nodes.threshold[u] = best_feat, best_bin
    nodes.left[u], nodes.right[u] = lc, rc

    stk_node.append(rc)
    stk_lo.append(mid)
    stk_hi.append(hi)
    stk_dep.append(dep + 1)

    stk_node.append(lc)
    stk_lo.append(lo)
    stk_hi.append(mid)
    stk_dep.append(dep + 1)


@par.jit
def fit_gbdt_bfs(
  X_bin,
  grads,
  hess,
  n_bins,
  max_depth,
  min_samples=4,
  reg_lambda=np.float32(1.0),
  min_child_weight=np.float32(1.0),
):
  """Implementation 1: pure level-wise BFS with histogram subtraction."""
  n_samples, n_features = X_bin.shape
  cfg = TreeConfig(n_features, n_bins, max_depth, min_samples, reg_lambda, min_child_weight)
  nodes = _alloc_tree_nodes(max_depth)
  max_nodes = len(nodes.feature)

  g_hists = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)
  h_hists = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)
  sample_ids = np.arange(n_samples, dtype=np.int64)
  scratch_ids = np.empty(n_samples, dtype=np.int64)

  bfs_nodes = np.array([0], dtype=np.int64)
  bfs_starts = np.array([0], dtype=np.int64)
  bfs_stops = np.array([n_samples], dtype=np.int64)
  bfs_built = np.array([0], dtype=np.int64)

  for depth in range(max_depth + 1):
    n_active = len(bfs_nodes)
    if n_active == 0:
      break

    next_nodes = np.empty(2 * n_active, dtype=np.int64)
    next_starts = np.empty(2 * n_active, dtype=np.int64)
    next_stops = np.empty(2 * n_active, dtype=np.int64)
    next_built = np.empty(2 * n_active, dtype=np.int64)
    n_next = 0

    for idx in range(n_active):
      u = bfs_nodes[idx]
      lo, hi = bfs_starts[idx], bfs_stops[idx]
      if bfs_built[idx] == 0:
        _build_node_hist_parallel(X_bin, grads, hess, sample_ids, lo, hi, g_hists[u], h_hists[u], cfg)

      nodes.value[u] = -np.sum(g_hists[u, 0]) / (np.sum(h_hists[u, 0]) + reg_lambda)
      if depth >= max_depth or (hi - lo) < min_samples:
        continue

      bf, bb, _ = _best_split_from_hist(g_hists[u], h_hists[u], cfg)
      if bf < 0:
        continue

      nl, nr = _partition_segment(X_bin, sample_ids, scratch_ids, lo, hi, bf, bb)
      if nl == 0 or nr == 0:
        continue

      mid = lo + nl
      lc, rc = 2 * u + 1, 2 * u + 2
      nodes.feature[u], nodes.threshold[u] = bf, bb
      nodes.left[u], nodes.right[u] = lc, rc

      built = 0
      if (depth + 1) < max_depth:
        small_c, large_c, s_lo, s_hi = (lc, rc, lo, mid) if nl <= nr else (rc, lc, mid, hi)
        _build_node_hist_parallel(X_bin, grads, hess, sample_ids, s_lo, s_hi, g_hists[small_c], h_hists[small_c], cfg)
        g_hists[large_c] = g_hists[u] - g_hists[small_c]
        h_hists[large_c] = h_hists[u] - h_hists[small_c]
        built = 1

      next_nodes[n_next], next_starts[n_next], next_stops[n_next], next_built[n_next] = lc, lo, mid, built
      next_nodes[n_next + 1], next_starts[n_next + 1], next_stops[n_next + 1], next_built[n_next + 1] = (
        rc,
        mid,
        hi,
        built,
      )
      n_next += 2

    bfs_nodes = next_nodes[:n_next]
    bfs_starts = next_starts[:n_next]
    bfs_stops = next_stops[:n_next]
    bfs_built = next_built[:n_next]

  return nodes.as_tuple()


@par.jit
def fit_cudatree_gbdt(
  X_bin,
  grads,
  hess,
  n_bins,
  max_depth,
  min_samples=4,
  reg_lambda=np.float32(1.0),
  min_child_weight=np.float32(1.0),
  dfs_thresh=64,
):
  """Implementation 2: adaptive BFS with histogram subtraction, handing small
  subtrees (``count <= dfs_thresh``) off to a single thread-per-subtree DFS pass.

  Returns ``(node_feat, node_thresh, node_left, node_right, node_value)`` indexed
  by heap node ID ``0 .. 2**(max_depth+1) - 2``.
  """
  n_samples, n_features = X_bin.shape
  cfg = TreeConfig(n_features, n_bins, max_depth, min_samples, reg_lambda, min_child_weight)
  nodes = _alloc_tree_nodes(max_depth)
  max_nodes = len(nodes.feature)

  g_hists = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)
  h_hists = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)
  sample_ids = np.arange(n_samples, dtype=np.int64)
  scratch_ids = np.empty(n_samples, dtype=np.int64)

  bfs_nodes = np.array([0], dtype=np.int64)
  bfs_starts = np.array([0], dtype=np.int64)
  bfs_stops = np.array([n_samples], dtype=np.int64)
  bfs_built = np.array([0], dtype=np.int64)

  dfs_nodes = np.empty(max_nodes, dtype=np.int64)
  dfs_starts = np.empty(max_nodes, dtype=np.int64)
  dfs_stops = np.empty(max_nodes, dtype=np.int64)
  dfs_depths = np.empty(max_nodes, dtype=np.int64)
  n_dfs = 0

  for depth in range(max_depth + 1):
    n_active = len(bfs_nodes)
    if n_active == 0:
      break

    next_nodes = np.empty(2 * n_active, dtype=np.int64)
    next_starts = np.empty(2 * n_active, dtype=np.int64)
    next_stops = np.empty(2 * n_active, dtype=np.int64)
    next_built = np.empty(2 * n_active, dtype=np.int64)
    n_next = 0

    for idx in range(n_active):
      u = bfs_nodes[idx]
      lo, hi = bfs_starts[idx], bfs_stops[idx]
      cnt = hi - lo

      if cnt <= dfs_thresh and 0 < depth < max_depth:
        dfs_nodes[n_dfs], dfs_starts[n_dfs], dfs_stops[n_dfs], dfs_depths[n_dfs] = u, lo, hi, depth
        n_dfs += 1
        continue

      if bfs_built[idx] == 0:
        _build_node_hist_parallel(X_bin, grads, hess, sample_ids, lo, hi, g_hists[u], h_hists[u], cfg)

      nodes.value[u] = -np.sum(g_hists[u, 0]) / (np.sum(h_hists[u, 0]) + reg_lambda)
      if depth >= max_depth or cnt < min_samples:
        continue

      bf, bb, _ = _best_split_from_hist(g_hists[u], h_hists[u], cfg)
      if bf < 0:
        continue

      nl, nr = _partition_segment(X_bin, sample_ids, scratch_ids, lo, hi, bf, bb)
      if nl == 0 or nr == 0:
        continue

      mid = lo + nl
      lc, rc = 2 * u + 1, 2 * u + 2
      nodes.feature[u], nodes.threshold[u] = bf, bb
      nodes.left[u], nodes.right[u] = lc, rc

      built = 0
      if (depth + 1) < max_depth and (nl > dfs_thresh or nr > dfs_thresh):
        small_c, large_c, s_lo, s_hi = (lc, rc, lo, mid) if nl <= nr else (rc, lc, mid, hi)
        _build_node_hist_parallel(X_bin, grads, hess, sample_ids, s_lo, s_hi, g_hists[small_c], h_hists[small_c], cfg)
        g_hists[large_c] = g_hists[u] - g_hists[small_c]
        h_hists[large_c] = h_hists[u] - h_hists[small_c]
        built = 1

      next_nodes[n_next], next_starts[n_next], next_stops[n_next], next_built[n_next] = lc, lo, mid, built
      next_nodes[n_next + 1], next_starts[n_next + 1], next_stops[n_next + 1], next_built[n_next + 1] = (
        rc,
        mid,
        hi,
        built,
      )
      n_next += 2

    bfs_nodes = next_nodes[:n_next]
    bfs_starts = next_starts[:n_next]
    bfs_stops = next_stops[:n_next]
    bfs_built = next_built[:n_next]

  if n_dfs > 0:
    par.parfor(
      lambda t: _dfs_complete_subtree(
        dfs_nodes[t],
        dfs_starts[t],
        dfs_stops[t],
        dfs_depths[t],
        X_bin,
        grads,
        hess,
        sample_ids,
        scratch_ids,
        cfg,
        nodes,
      ),
      n_dfs,
    )

  return nodes.as_tuple()


@par.jit
def predict_cudatree_gbdt(X_bin, node_feat, node_thresh, node_left, node_right, node_value):
  """Predict leaf values for every row in parallel."""

  def predict_row(row):
    u = 0
    while node_feat[u] >= 0:
      u = node_left[u] if row[node_feat[u]] <= node_thresh[u] else node_right[u]
    return node_value[u]

  return np.array([predict_row(row) for row in X_bin])


def main():
  rng = np.random.default_rng(0)
  n_samples, n_features, n_bins = 20000, 16, 32
  X_bin = rng.integers(0, n_bins, size=(n_samples, n_features), dtype=np.int32)
  y = (
    0.5 * (X_bin[:, 0] > 15).astype(np.float32)
    - 0.3 * (X_bin[:, 3] < 8).astype(np.float32)
    + 0.2 * ((X_bin[:, 1] > 20) & (X_bin[:, 2] < 12)).astype(np.float32)
    + 0.05 * rng.standard_normal(n_samples).astype(np.float32)
  )
  grads = -y
  hess = np.ones(n_samples, dtype=np.float32)

  bfs_tree = fit_gbdt_bfs(X_bin, grads, hess, n_bins, 6)
  adapt_tree = fit_cudatree_gbdt(X_bin, grads, hess, n_bins, 6)
  for a, b in zip(bfs_tree[:4], adapt_tree[:4]):
    np.testing.assert_array_equal(a, b)
  np.testing.assert_allclose(bfs_tree[4], adapt_tree[4], rtol=1e-4, atol=1e-4)

  t0 = time.perf_counter()
  for _ in range(5):
    adapt_tree = fit_cudatree_gbdt(X_bin, grads, hess, n_bins, 6)
  dt = (time.perf_counter() - t0) / 5
  preds = predict_cudatree_gbdt(X_bin, *adapt_tree)
  mse_before = float(np.mean(y * y))
  mse_after = float(np.mean((y - preds) ** 2))
  assert mse_after < 0.25 * mse_before, (mse_before, mse_after)
  n_splits = int(np.sum(adapt_tree[0] >= 0))
  print(
    f"GBDT (N={n_samples}, F={n_features}, B={n_bins}, depth=6): "
    f"{n_splits} splits, MSE {mse_before:.4f} -> {mse_after:.4f}, {dt * 1e3:.2f} ms/tree"
  )


if __name__ == "__main__":
  main()
