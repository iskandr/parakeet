"""Benchmark histogram GBDT tree builders: pure BFS vs. adaptive BFS-to-DFS vs. libraries.

Compares both Parakeet tree-building implementations from ``examples/cudatree_gbdt.py``
(``fit_gbdt_bfs`` and ``fit_cudatree_gbdt``) on single-core and multicore backends
against installed tree libraries (``scikit-learn``, ``lightgbm``, ``xgboost``) on
balanced and skewed datasets.

Run::

    python -m benchmarks.tree_models
"""

from __future__ import annotations

import time

import numpy as np

from examples.cudatree_gbdt import fit_cudatree_gbdt, fit_gbdt_bfs, predict_cudatree_gbdt


def _time_reps(fn, *, reps: int = 7, warmup: int = 2) -> tuple[float, float]:
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


def make_dataset(n_samples: int, n_features: int, n_bins: int, *, skewed: bool = False, seed: int = 0):
  rng = np.random.default_rng(seed)
  if skewed:
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
  return X_bin, y, -y, np.ones(n_samples, dtype=np.float32)


def run_benchmark(n_samples: int = 50_000, n_features: int = 16, n_bins: int = 32, max_depth: int = 6, reps: int = 5):
  for label, skewed in (("balanced", False), ("skewed", True)):
    X_bin, y, grads, hess = make_dataset(n_samples, n_features, n_bins, skewed=skewed)
    print(f"\n=== Dataset: {label} (N={n_samples}, F={n_features}, B={n_bins}, depth={max_depth}) ===")
    for impl_name, fit_fn in (("pure BFS", fit_gbdt_bfs), ("BFS->DFS", fit_cudatree_gbdt)):
      for backend in ("c", "multicore"):
        med, iqr = _time_reps(
          lambda f=fit_fn, b=backend: f(X_bin, grads, hess, n_bins, max_depth, _backend=b), reps=reps
        )
        tree = fit_fn(X_bin, grads, hess, n_bins, max_depth, _backend=backend)
        preds = predict_cudatree_gbdt(X_bin, *tree, _backend=backend)
        mse = float(np.mean((y - preds) ** 2))
        print(f"  Parakeet {impl_name:8s} ({backend:9s}): median={med:7.2f} ms  IQR={iqr:5.2f} ms  MSE={mse:.4f}")

    X_f = X_bin.astype(np.float32)
    try:
      from sklearn.ensemble import HistGradientBoostingRegressor

      est = HistGradientBoostingRegressor(max_iter=1, max_depth=max_depth, max_bins=n_bins, l2_regularization=1.0)
      med, iqr = _time_reps(lambda: est.fit(X_f, y), reps=reps)
      print(f"  scikit-learn HistGB           : median={med:7.2f} ms  IQR={iqr:5.2f} ms")
    except ImportError:
      pass

    try:
      import lightgbm as lgb

      ds = lgb.Dataset(X_f, label=y, free_raw_data=False)
      params = {"objective": "regression", "max_depth": max_depth, "max_bin": n_bins, "verbose": -1}
      med, iqr = _time_reps(lambda: lgb.train(params, ds, num_boost_round=1), reps=reps)
      print(f"  LightGBM (1 tree)             : median={med:7.2f} ms  IQR={iqr:5.2f} ms")
    except ImportError:
      pass

    try:
      import xgboost as xgb

      dmat = xgb.DMatrix(X_f, label=y)
      params = {"tree_method": "hist", "max_depth": max_depth, "max_bin": n_bins, "verbosity": 0}
      med, iqr = _time_reps(lambda: xgb.train(params, dmat, num_boost_round=1), reps=reps)
      print(f"  XGBoost hist (1 tree)         : median={med:7.2f} ms  IQR={iqr:5.2f} ms")
    except ImportError:
      pass


if __name__ == "__main__":
  run_benchmark()
