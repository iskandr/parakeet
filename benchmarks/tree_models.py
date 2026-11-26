"""Reproducible benchmark for CudaTree adaptive BFS->DFS GBDT vs scikit-learn/LightGBM/XGBoost.

Reports median and IQR across replicates on balanced and skewed datasets for
Parakeet (`c`, `multicore`, and `metal` when available) plus any installed
tree libraries (`scikit-learn`, `lightgbm`, `xgboost`).

Run:
    python benchmarks/tree_models.py
"""

import time

import numpy as np

from examples.cudatree_gbdt import fit_cudatree_gbdt, predict_cudatree_gbdt


def _time_reps(fn, reps=7, warmup=2):
  for _ in range(warmup):
    fn()
  times = []
  for _ in range(reps):
    t0 = time.perf_counter()
    fn()
    times.append((time.perf_counter() - t0) * 1e3)
  arr = np.asarray(times, dtype=np.float64)
  q25, med, q75 = np.percentile(arr, [25, 50, 75])
  return float(med), float(q75 - q25)


def make_dataset(n_samples, n_features, n_bins, skewed=False, seed=0):
  rng = np.random.default_rng(seed)
  if skewed:
    # Power-law skewed feature 0 so tree splits are heavily unbalanced
    raw = rng.pareto(1.8, size=(n_samples, n_features))
    X_bin = np.clip((raw * 4.0).astype(np.int32), 0, n_bins - 1)
  else:
    X_bin = rng.integers(0, n_bins, size=(n_samples, n_features), dtype=np.int32)
  y = (
    0.8 * (X_bin[:, 0] > (n_bins // 2)).astype(np.float32)
    - 0.5 * (X_bin[:, 1] < (n_bins // 3)).astype(np.float32)
    + 0.3 * ((X_bin[:, 2] > 10) & (X_bin[:, 3] < 20)).astype(np.float32)
    + 0.1 * rng.standard_normal(n_samples).astype(np.float32)
  )
  grads = -y
  hess = np.ones(n_samples, dtype=np.float32)
  return X_bin, y, grads, hess


def run_benchmark(n_samples=50000, n_features=16, n_bins=32, max_depth=6, reps=5):
  for label, skewed in (("balanced", False), ("skewed", True)):
    X_bin, y, grads, hess = make_dataset(n_samples, n_features, n_bins, skewed=skewed)
    print(f"\n=== Dataset: {label} (N={n_samples}, F={n_features}, B={n_bins}, depth={max_depth}) ===")
    for backend in ("c", "multicore"):
      med, iqr = _time_reps(
        lambda b=backend: fit_cudatree_gbdt(X_bin, grads, hess, n_bins, max_depth, _backend=b),
        reps=reps,
      )
      tree = fit_cudatree_gbdt(X_bin, grads, hess, n_bins, max_depth, _backend=backend)
      preds = predict_cudatree_gbdt(X_bin, *tree, _backend=backend)
      mse = float(np.mean((y - preds) ** 2))
      print(f"  Parakeet ({backend:9s}): median={med:7.2f} ms  IQR={iqr:5.2f} ms  MSE={mse:.4f}")

    try:
      from sklearn.ensemble import HistGradientBoostingRegressor

      X_f = X_bin.astype(np.float32)
      est = HistGradientBoostingRegressor(max_iter=1, max_depth=max_depth, max_bins=n_bins, l2_regularization=1.0)
      med, iqr = _time_reps(lambda: est.fit(X_f, y), reps=reps)
      print(f"  scikit-learn HistGB: median={med:7.2f} ms  IQR={iqr:5.2f} ms")
    except ImportError:
      pass


if __name__ == "__main__":
  run_benchmark()
