"""K-means clustering in four styles: ``_numpy``, ``_comp``, ``_loop``, and ``_par``."""

import numpy as np

import parakeet as par


def sqr_dist(x, y):
  return np.sum((x - y) ** 2)


@par.jit
def kmeans_numpy(X, k, niters=10):
  C = X[:k, :]
  for _ in range(niters):
    d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    A = np.argmin(d2, axis=1)
    C = np.array([np.mean(X[A == i, :], axis=0) for i in range(k)])
  return C


@par.jit
def kmeans_comp(X, k, niters=10):
  C = X[:k, :]
  for _ in range(niters):
    A = np.array([np.argmin([sqr_dist(x, c) for c in C]) for x in X])
    C = np.array([np.mean(X[A == i, :], axis=0) for i in range(k)])
  return C


@par.jit
def kmeans_loop(X, k, niters=10):
  n, d = X.shape
  C = X[:k, :].copy()
  for _ in range(niters):
    sums = np.zeros((k, d), dtype=X.dtype)
    counts = np.zeros(k, dtype=np.int64)
    for i in range(n):
      best_c = 0
      best_d = sqr_dist(X[i], C[0])
      for c in range(1, k):
        dist = sqr_dist(X[i], C[c])
        if dist < best_d:
          best_d = dist
          best_c = c
      sums[best_c] += X[i]
      counts[best_c] += 1
    C = sums / counts[:, None]
  return C


@par.jit
def kmeans_par(X, k, niters=10):
  C = X[:k, :]
  ones = np.ones(len(X), dtype=X.dtype)
  for _ in range(niters):
    A = np.argmin(par.allpairs(sqr_dist, X, C), axis=1)
    sums = par.map(lambda col: par.scatter_add(k, A, col), X.T).T
    counts = par.scatter_add(k, A, ones)
    C = sums / counts[:, None]
  return C


kmeans = kmeans_comp


def main(n: int = 2000, d: int = 16, k: int = 4):
  rng = np.random.default_rng(1)
  centers = rng.normal(size=(k, d)) * 5
  X = centers[rng.integers(0, k, n)] + rng.normal(size=(n, d))
  ref = kmeans_numpy(X, k, 15, _backend="python")
  for fn in (kmeans_numpy, kmeans_comp, kmeans_loop, kmeans_par):
    assert np.allclose(fn(X, k, 15), ref), fn.__name__
  return ref


if __name__ == "__main__":
  print(main())
