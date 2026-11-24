"""Compare exact pairwise reductions on one CPU, with KeOps installed separately.

OMP_NUM_THREADS=8 python -m benchmarks.keops_pairwise --threads 8 --repeats 25
On macOS the KeOps extension may require Homebrew libomp to be preloaded.
"""

import argparse
import os
import platform
import statistics
import time

import numpy as np

import parakeet as pk
from examples.algebraic_gallery import exact_1nn, rbf_kernel_regression


@pk.jit
def numpy_1nn(x, y):
  return np.argmin(np.sum((x[:, None, :] - y[None, :, :]) ** np.float32(2), axis=2), axis=1)


def run_case(n, m, d, rng, repeats):
  from pykeops.numpy import Genred

  x = rng.standard_normal((n, d), dtype=np.float32)
  y = rng.standard_normal((m, d), dtype=np.float32)
  v = rng.standard_normal((m, 4), dtype=np.float32)
  variables = [f"X=Vi({d})", f"Y=Vj({d})"]
  nn = Genred("SqDist(X,Y)", variables, reduction_op="ArgMin", axis=1)
  rbf = Genred("-SqDist(X,Y)", variables + ["V=Vj(4)"], reduction_op="SumSoftMaxWeight", formula2="V", axis=1)

  # Bound reference memory to one row of distances, and accumulate in float64.
  nearest = np.empty(n, dtype=np.int64)
  gaussian = np.empty((n, 4))
  y64, v64 = y.astype(np.float64), v.astype(np.float64)
  for i, row in enumerate(x):
    dist = np.sum((y64 - row) ** 2, axis=1)
    nearest[i] = np.argmin(dist)
    weights = np.exp(np.min(dist) - dist)
    gaussian[i] = (weights / weights.sum()) @ v64

  functions = {
    "1nn/dsl": lambda: exact_1nn(x, y),
    "1nn/numpy-jit": lambda: numpy_1nn(x, y),
    "1nn/keops": lambda: nn(x, y, backend="CPU").ravel(),
    "rbf/numpy-jit": lambda: rbf_kernel_regression(x, y, v, np.float32(-1)),
    "rbf/keops": lambda: rbf(x, y, v, backend="CPU"),
  }
  for name, call in functions.items():
    result = call()
    if name.startswith("1nn"):
      np.testing.assert_array_equal(result, nearest)
    else:
      np.testing.assert_allclose(result, gaussian, rtol=5e-4, atol=2e-5)
  samples = {name: [] for name in functions}
  for _ in range(repeats):
    for name in rng.permutation(list(functions)):
      start = time.perf_counter_ns()
      functions[name]()
      samples[name].append((time.perf_counter_ns() - start) / 1e6)
  print(n, m, d, *(f"{statistics.median(v):.3f}" for v in samples.values()), flush=True)


def main():
  import pykeops

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
  parser.add_argument("--repeats", type=int, default=25)
  args = parser.parse_args()
  if os.environ.get("OMP_NUM_THREADS") != str(args.threads) or min(args.repeats, args.threads) <= 0:
    parser.error("Set OMP_NUM_THREADS equal to --threads and use positive counts.")
  print(f"{platform.platform()}, NumPy {np.__version__}, PyKeOps {pykeops.__version__}, {args.threads} threads")
  print("median ms: N M D 1nn/dsl 1nn/numpy-jit 1nn/keops rbf/numpy-jit rbf/keops")
  rng = np.random.default_rng(123)
  with pk.options(backend="threads", blas=False, num_threads=args.threads):
    for n, m, d in [(256, 1024, 3), (512, 2048, 16), (1024, 4096, 32)]:
      run_case(n, m, d, rng, args.repeats)


if __name__ == "__main__":
  main()
