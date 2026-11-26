"""All-pairs squared distances on a GPU (Metal on Apple Silicon, CUDA on NVIDIA).

GPUs want float32: Apple GPUs do not support float64 at all, in which case
Parakeet warns and runs the loop on the CPU instead.
"""

import sys

import numpy as np

import parakeet
import parakeet as par


def sqr_dist(a, b):
  return np.sum((a - b) ** 2)


def all_distances(X, C):
  return parakeet.allpairs(sqr_dist, X, C)


def pick_backend() -> str:
  available = parakeet.available_backends()
  for name in ("cuda", "metal"):
    if available.get(name) == "ok":
      return name
  return "multicore"


def main(n: int = 20_000, k: int = 64, d: int = 32, backend: str | None = None):
  backend = backend or pick_backend()
  rng = np.random.default_rng(7)
  X = rng.random((n, d), dtype=np.float32)
  C = rng.random((k, d), dtype=np.float32)
  D = par.jit(all_distances, backend=backend)(X, C)
  ref = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
  assert D.dtype == np.float32 and np.allclose(D, ref, rtol=1e-3, atol=1e-3)
  return backend, D


if __name__ == "__main__":
  name, D = main(backend=sys.argv[1] if len(sys.argv) > 1 else None)
  print(f"computed {D.shape} distances on backend {name!r}")
