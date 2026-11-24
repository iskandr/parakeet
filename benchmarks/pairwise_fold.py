"""Exact 1-NN scaling, with fixed query count and growing reference count.

OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 python -m benchmarks.pairwise_fold --repeats 25
Requires the optional PyKeOps and faiss-cpu packages. On macOS, if KeOps needs
libomp preloaded, use the libomp shipped with the Faiss wheel for both libraries.
"""

import argparse
import os
import platform
import statistics
import time

import numpy as np

import parakeet as pk
from benchmarks.keops_pairwise import exact_1nn, numpy_1nn


def main():
  import faiss
  import pykeops
  from pykeops.numpy import Genred

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repeats", type=int, default=25)
  parser.add_argument("--references", type=int, nargs="+", default=[1024, 4096, 16384])
  args = parser.parse_args()
  threads = int(os.environ["OMP_NUM_THREADS"])
  if min(threads, args.repeats, *args.references) <= 0:
    parser.error("Use positive sizes, repeat count and OMP_NUM_THREADS.")
  faiss.omp_set_num_threads(threads)
  rng = np.random.default_rng(71)
  n, d = 1024, 32
  x = rng.standard_normal((n, d), dtype=np.float32)
  keops = Genred("SqDist(X,Y)", [f"X=Vi({d})", f"Y=Vj({d})"], reduction_op="ArgMin", axis=1)
  print(
    platform.platform(),
    f"{threads} threads; NumPy {np.__version__}, PyKeOps {pykeops.__version__}, Faiss {faiss.__version__}",
  )
  print("N M D | median ms: DSL off/on, NumPy off/on, KeOps, Faiss search | Faiss build ms", flush=True)
  with pk.options(backend="threads", num_threads=threads, blas=False):
    for m in args.references:
      y = rng.standard_normal((m, d), dtype=np.float32)
      y64 = y.astype(np.float64)
      reference = np.array([np.argmin(np.sum((y64 - row) ** 2, axis=1)) for row in x])
      start = time.perf_counter_ns()
      index = faiss.IndexFlatL2(d)
      index.add(y)
      build = (time.perf_counter_ns() - start) / 1e6
      old = ("pairwise fold tiling",)
      runs = [
        (lambda: exact_1nn(x, y), old),
        (lambda: exact_1nn(x, y), ()),
        (lambda: numpy_1nn(x, y), old),
        (lambda: numpy_1nn(x, y), ()),
        (lambda: keops(x, y, backend="CPU").ravel(), ()),
        (lambda: index.search(x, 1)[1].ravel(), ()),
      ]
      times = [[] for _ in runs]
      for i in range(args.repeats + 1):
        for j in rng.permutation(len(runs)):
          call, skip = runs[j]
          with pk.options(skip_transforms=skip):
            start = time.perf_counter_ns()
            result = call()
            elapsed = (time.perf_counter_ns() - start) / 1e6
          if i:
            times[j].append(elapsed)
          else:
            np.testing.assert_array_equal(result, reference)
      print(n, m, d, *(f"{statistics.median(t):.3f}" for t in times), f"{build:.3f}", flush=True)


if __name__ == "__main__":
  main()
