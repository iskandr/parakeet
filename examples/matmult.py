"""Matrix multiplication in Parakeet's four standard styles:
``_numpy``, ``_comp``, ``_loop``, and ``_par``.
"""

import numpy as np

import parakeet as par


@par.jit
def matmult_numpy(X, Y):
  return X @ Y


@par.jit
def matmult_comp(X, Y):
  return np.array([[np.dot(x, y) for y in Y.T] for x in X])


@par.jit
def matmult_loop(X, Y):
  out = np.zeros((X.shape[0], Y.shape[1]), dtype=X.dtype)
  for i in range(X.shape[0]):
    for j in range(Y.shape[1]):
      for k in range(X.shape[1]):
        out[i, j] += X[i, k] * Y[k, j]
  return out


@par.jit
def matmult_par(X, Y):
  return par.allpairs(np.dot, X, Y.T)


def main(n: int = 120):
  rng = np.random.default_rng(5)
  X, Y = rng.random((n, n + 3)), rng.random((n + 3, n - 2))
  ref = X @ Y
  for fn in (matmult_numpy, matmult_comp, matmult_loop, matmult_par):
    assert np.allclose(fn(X, Y), ref), fn.__name__
  return ref


if __name__ == "__main__":
  print(main().shape)
