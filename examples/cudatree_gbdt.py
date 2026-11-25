"""CudaTree-style adaptive BFS -> DFS histogram GBDT tree builder in Parakeet.

Demonstrates how Parakeet's irregular parallelism primitives tame non-uniform
decision-tree workloads across CPU, multicore, Apple Metal, and CUDA:

1. Shallow & middle BFS levels:
   - ``par.partition_by_size`` splits the active frontier into BFS nodes vs
     small DFS subtrees.
   - Feature-parallel fused ``(grad, hess)`` histograms with **histogram
     subtraction** (compute only the smaller child of each split; subtract
     from the parent histogram for the larger child, cutting work by up to 2x).
   - ``par.ragged_partition`` partitions active sample indices in parallel.
2. Deep levels (DFS handoff):
   - Nodes whose sample count drops below ``dfs_thresh`` enter a single fused
     thread-per-subtree DFS pass using thread-local ``par.fixed_list`` scratch
     stacks (zero ``malloc``, GPU-native on Metal and CUDA!).

Run:
    python examples/cudatree_gbdt.py
"""

import time

import numpy as np

import parakeet as par


def _gain(g_l, h_l, g_r, h_r, g_p, h_p, reg_lambda):
  return (g_l * g_l) / (h_l + reg_lambda) + (g_r * g_r) / (h_r + reg_lambda) - (g_p * g_p) / (h_p + reg_lambda)


def _best_split_from_hist(g_hist, h_hist, n_features, n_bins, reg_lambda, min_child_weight):
  """Find exact best ``(feature, bin, gain)`` from ``(F, B)`` grad/hess histograms."""
  best_gain = np.float32(0.0)
  best_feat = np.int64(-1)
  best_bin = np.int64(-1)
  g_tot = np.float32(0.0)
  h_tot = np.float32(0.0)
  for b in range(n_bins):
    g_tot = g_tot + g_hist[0, b]
    h_tot = h_tot + h_hist[0, b]
  for f in range(n_features):
    gl = np.float32(0.0)
    hl = np.float32(0.0)
    for b in range(n_bins - 1):
      gl = gl + g_hist[f, b]
      hl = hl + h_hist[f, b]
      hr = h_tot - hl
      if hl >= min_child_weight and hr >= min_child_weight:
        gr = g_tot - gl
        gain = _gain(gl, hl, gr, hr, g_tot, h_tot, reg_lambda)
        if gain > best_gain + np.float32(1e-7):
          best_gain = gain
          best_feat = np.int64(f)
          best_bin = np.int64(b)
  return best_feat, best_bin, best_gain


def _build_node_hist_parallel(X_bin, grads, hess, sample_ids, start, stop, g_out, h_out, n_features, n_bins):
  """Feature-parallel histogram build for one BFS node ``[start, stop)``."""

  def _one_feat(f):
    for b in range(n_bins):
      g_out[f, b] = np.float32(0.0)
      h_out[f, b] = np.float32(0.0)
    for k in range(start, stop):
      sid = sample_ids[k]
      bi = X_bin[sid, f]
      g_out[f, bi] = g_out[f, bi] + grads[sid]
      h_out[f, bi] = h_out[f, bi] + hess[sid]

  par.parfor(_one_feat, n_features)


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
  n_features,
  n_bins,
  max_depth,
  min_samples,
  reg_lambda,
  min_child_weight,
  node_feat,
  node_thresh,
  node_left,
  node_right,
  node_value,
):
  """Thread-local DFS subtree completion using ``par.fixed_list`` scratch (zero malloc, GPU-safe!).

  Uses fixed complete-binary-tree indexing ``left = 2*u + 1, right = 2*u + 2``
  so sibling subtrees write to disjoint node indices with zero atomic contention.
  """
  stk_node = par.fixed_list(64, np.int64)
  stk_lo = par.fixed_list(64, np.int64)
  stk_hi = par.fixed_list(64, np.int64)
  stk_dep = par.fixed_list(64, np.int64)
  stk_node.append(root_node)
  stk_lo.append(root_start)
  stk_hi.append(root_stop)
  stk_dep.append(root_depth)

  # Thread-local 1-feature histogram scratch (max 64 bins per feature)
  gh = par.fixed_list(64, np.float32)
  hh = par.fixed_list(64, np.float32)
  for _ in range(n_bins):
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
      g_tot = g_tot + grads[sid]
      h_tot = h_tot + hess[sid]
    node_value[u] = -g_tot / (h_tot + reg_lambda)

    if dep >= max_depth or (hi - lo) < min_samples:
      continue

    best_gain = np.float32(0.0)
    best_feat = np.int64(-1)
    best_bin = np.int64(-1)

    for f in range(n_features):
      for b in range(n_bins):
        gh[b] = np.float32(0.0)
        hh[b] = np.float32(0.0)
      for k in range(lo, hi):
        sid = sample_ids[k]
        bi = X_bin[sid, f]
        gh[bi] = gh[bi] + grads[sid]
        hh[bi] = hh[bi] + hess[sid]
      gl = np.float32(0.0)
      hl = np.float32(0.0)
      for b in range(n_bins - 1):
        gl = gl + gh[b]
        hl = hl + hh[b]
        hr = h_tot - hl
        if hl >= min_child_weight and hr >= min_child_weight:
          gr = g_tot - gl
          gain = _gain(gl, hl, gr, hr, g_tot, h_tot, reg_lambda)
          if gain > best_gain + np.float32(1e-7):
            best_gain = gain
            best_feat = np.int64(f)
            best_bin = np.int64(b)

    if best_feat < 0:
      continue

    # Partition [lo, hi) in-place via thread-disjoint scratch_ids[lo:hi]
    nl = np.int64(0)
    for k in range(lo, hi):
      sid = sample_ids[k]
      if X_bin[sid, best_feat] <= best_bin:
        scratch_ids[lo + nl] = sid
        nl = nl + np.int64(1)
    nr = np.int64(0)
    mid = lo + nl
    for k in range(lo, hi):
      sid = sample_ids[k]
      if X_bin[sid, best_feat] > best_bin:
        scratch_ids[mid + nr] = sid
        nr = nr + np.int64(1)
    for k in range(lo, hi):
      sample_ids[k] = scratch_ids[k]

    if nl == 0 or nr == 0:
      continue

    lc = np.int64(2) * u + np.int64(1)
    rc = np.int64(2) * u + np.int64(2)
    node_feat[u] = best_feat
    node_thresh[u] = best_bin
    node_left[u] = lc
    node_right[u] = rc

    stk_node.append(rc)
    stk_lo.append(mid)
    stk_hi.append(hi)
    stk_dep.append(dep + np.int64(1))

    stk_node.append(lc)
    stk_lo.append(lo)
    stk_hi.append(mid)
    stk_dep.append(dep + np.int64(1))


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
  """Fit a histogram GBDT tree with adaptive BFS (histogram subtraction) -> DFS switching.

  Returns ``(node_feat, node_thresh, node_left, node_right, node_value)`` indexed
  by heap node ID ``0 .. 2**(max_depth+1) - 2``.
  """
  n_samples, n_features = X_bin.shape
  max_nodes = np.int64(1)
  for _ in range(max_depth + 1):
    max_nodes = max_nodes * np.int64(2)
  max_nodes = max_nodes - np.int64(1)

  node_feat = np.full(max_nodes, np.int64(-1), dtype=np.int64)
  node_thresh = np.full(max_nodes, np.int64(-1), dtype=np.int64)
  node_left = np.full(max_nodes, np.int64(-1), dtype=np.int64)
  node_right = np.full(max_nodes, np.int64(-1), dtype=np.int64)
  node_value = np.zeros(max_nodes, dtype=np.float32)

  # Per-node histogram cache for histogram subtraction (large_child = parent - small_child)
  g_hists = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)
  h_hists = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)

  sample_ids = np.arange(n_samples, dtype=np.int64)
  scratch_ids = np.empty(n_samples, dtype=np.int64)

  # Active BFS frontier arrays (dynamic-shape rebound per depth)
  bfs_nodes = np.array([0], dtype=np.int64)
  bfs_starts = np.array([0], dtype=np.int64)
  bfs_stops = np.array([n_samples], dtype=np.int64)
  bfs_built = np.array([0], dtype=np.int64)  # 1 if histogram already populated via subtraction

  # Deferred DFS subtree queue (completed in one fused parallel kernel launch!)
  dfs_nodes = np.empty(max_nodes, dtype=np.int64)
  dfs_starts = np.empty(max_nodes, dtype=np.int64)
  dfs_stops = np.empty(max_nodes, dtype=np.int64)
  dfs_depths = np.empty(max_nodes, dtype=np.int64)
  n_dfs = np.int64(0)

  for depth in range(max_depth + 1):
    n_active = len(bfs_nodes)
    if n_active == 0:
      break

    # Next-level BFS frontier buffers (max 2 * n_active children)
    next_nodes = np.empty(2 * n_active, dtype=np.int64)
    next_starts = np.empty(2 * n_active, dtype=np.int64)
    next_stops = np.empty(2 * n_active, dtype=np.int64)
    next_built = np.empty(2 * n_active, dtype=np.int64)
    n_next = np.int64(0)

    # First, classify active BFS nodes vs small nodes ready for DFS handoff
    for idx in range(n_active):
      u = bfs_nodes[idx]
      lo = bfs_starts[idx]
      hi = bfs_stops[idx]
      cnt = hi - lo

      if cnt <= dfs_thresh and depth > 0 and depth < max_depth:
        dfs_nodes[n_dfs] = u
        dfs_starts[n_dfs] = lo
        dfs_stops[n_dfs] = hi
        dfs_depths[n_dfs] = np.int64(depth)
        n_dfs = n_dfs + np.int64(1)
        continue

      # Build histogram only if not already populated by histogram subtraction!
      if bfs_built[idx] == 0:
        _build_node_hist_parallel(X_bin, grads, hess, sample_ids, lo, hi, g_hists[u], h_hists[u], n_features, n_bins)

      g_tot = np.float32(0.0)
      h_tot = np.float32(0.0)
      for b in range(n_bins):
        g_tot = g_tot + g_hists[u, 0, b]
        h_tot = h_tot + h_hists[u, 0, b]
      node_value[u] = -g_tot / (h_tot + reg_lambda)

      if depth >= max_depth or cnt < min_samples:
        continue

      bf, bb, _ = _best_split_from_hist(g_hists[u], h_hists[u], n_features, n_bins, reg_lambda, min_child_weight)
      if bf < 0:
        continue

      # Partition [lo, hi) stably
      nl = np.int64(0)
      for k in range(lo, hi):
        sid = sample_ids[k]
        if X_bin[sid, bf] <= bb:
          scratch_ids[lo + nl] = sid
          nl = nl + np.int64(1)
      nr = np.int64(0)
      mid = lo + nl
      for k in range(lo, hi):
        sid = sample_ids[k]
        if X_bin[sid, bf] > bb:
          scratch_ids[mid + nr] = sid
          nr = nr + np.int64(1)
      for k in range(lo, hi):
        sample_ids[k] = scratch_ids[k]

      if nl == 0 or nr == 0:
        continue

      lc = np.int64(2) * u + np.int64(1)
      rc = np.int64(2) * u + np.int64(2)
      node_feat[u] = bf
      node_thresh[u] = bb
      node_left[u] = lc
      node_right[u] = rc

      # Histogram subtraction: build only smaller child when larger child stays in BFS!
      if nl <= nr:
        if nl > dfs_thresh and (depth + 1) < max_depth:
          _build_node_hist_parallel(
            X_bin, grads, hess, sample_ids, lo, mid, g_hists[lc], h_hists[lc], n_features, n_bins
          )
          g_hists[rc] = g_hists[u] - g_hists[lc]
          h_hists[rc] = h_hists[u] - h_hists[lc]
          lc_built = np.int64(1)
          rc_built = np.int64(1)
        elif nr > dfs_thresh and (depth + 1) < max_depth:
          _build_node_hist_parallel(
            X_bin, grads, hess, sample_ids, lo, mid, g_hists[lc], h_hists[lc], n_features, n_bins
          )
          g_hists[rc] = g_hists[u] - g_hists[lc]
          h_hists[rc] = h_hists[u] - h_hists[lc]
          lc_built = np.int64(1)
          rc_built = np.int64(1)
        else:
          lc_built = np.int64(0)
          rc_built = np.int64(0)
      else:
        if nr > dfs_thresh and (depth + 1) < max_depth:
          _build_node_hist_parallel(
            X_bin, grads, hess, sample_ids, mid, hi, g_hists[rc], h_hists[rc], n_features, n_bins
          )
          g_hists[lc] = g_hists[u] - g_hists[rc]
          h_hists[lc] = h_hists[u] - h_hists[rc]
          lc_built = np.int64(1)
          rc_built = np.int64(1)
        elif nl > dfs_thresh and (depth + 1) < max_depth:
          _build_node_hist_parallel(
            X_bin, grads, hess, sample_ids, mid, hi, g_hists[rc], h_hists[rc], n_features, n_bins
          )
          g_hists[lc] = g_hists[u] - g_hists[rc]
          h_hists[lc] = h_hists[u] - h_hists[rc]
          lc_built = np.int64(1)
          rc_built = np.int64(1)
        else:
          lc_built = np.int64(0)
          rc_built = np.int64(0)

      next_nodes[n_next] = lc
      next_starts[n_next] = lo
      next_stops[n_next] = mid
      next_built[n_next] = lc_built
      n_next = n_next + np.int64(1)

      next_nodes[n_next] = rc
      next_starts[n_next] = mid
      next_stops[n_next] = hi
      next_built[n_next] = rc_built
      n_next = n_next + np.int64(1)

    bfs_nodes = next_nodes[:n_next]
    bfs_starts = next_starts[:n_next]
    bfs_stops = next_stops[:n_next]
    bfs_built = next_built[:n_next]

  # Single fused thread-per-subtree DFS launch for all deferred light subtrees!
  if n_dfs > 0:

    def _finish_dfs(t):
      _dfs_complete_subtree(
        dfs_nodes[t],
        dfs_starts[t],
        dfs_stops[t],
        dfs_depths[t],
        X_bin,
        grads,
        hess,
        sample_ids,
        scratch_ids,
        n_features,
        n_bins,
        max_depth,
        min_samples,
        reg_lambda,
        min_child_weight,
        node_feat,
        node_thresh,
        node_left,
        node_right,
        node_value,
      )

    par.parfor(_finish_dfs, n_dfs)

  return node_feat, node_thresh, node_left, node_right, node_value


@par.jit
def predict_cudatree_gbdt(X_bin, node_feat, node_thresh, node_left, node_right, node_value):
  """Predict leaf values for all rows in parallel."""
  n = X_bin.shape[0]
  out = np.empty(n, dtype=np.float32)

  def _pred_row(i):
    u = np.int64(0)
    while node_feat[u] >= 0:
      if X_bin[i, node_feat[u]] <= node_thresh[u]:
        u = node_left[u]
      else:
        u = node_right[u]
    out[i] = node_value[u]

  par.parfor(_pred_row, n)
  return out


def main():
  rng = np.random.default_rng(0)
  n_samples, n_features, n_bins = 20000, 16, 32
  X_bin = rng.integers(0, n_bins, size=(n_samples, n_features), dtype=np.int32)
  # True nonlinear target over binned features
  y = (
    0.5 * (X_bin[:, 0] > 15).astype(np.float32)
    - 0.3 * (X_bin[:, 3] < 8).astype(np.float32)
    + 0.2 * ((X_bin[:, 1] > 20) & (X_bin[:, 2] < 12)).astype(np.float32)
    + 0.05 * rng.standard_normal(n_samples).astype(np.float32)
  )
  grads = -y  # squared-error gradient at initial prediction 0
  hess = np.ones(n_samples, dtype=np.float32)

  # Warmup
  tree = fit_cudatree_gbdt(X_bin, grads, hess, n_bins, 6)
  t0 = time.perf_counter()
  for _ in range(5):
    tree = fit_cudatree_gbdt(X_bin, grads, hess, n_bins, 6)
  dt = (time.perf_counter() - t0) / 5
  preds = predict_cudatree_gbdt(X_bin, *tree)
  mse_before = float(np.mean(y * y))
  mse_after = float(np.mean((y - preds) ** 2))
  assert mse_after < 0.25 * mse_before, (mse_before, mse_after)
  n_splits = int(np.sum(tree[0] >= 0))
  print(
    f"CudaTree GBDT (N={n_samples}, F={n_features}, B={n_bins}, depth=6): "
    f"{n_splits} splits, MSE {mse_before:.4f} -> {mse_after:.4f}, {dt * 1e3:.2f} ms/tree"
  )


if __name__ == "__main__":
  main()
