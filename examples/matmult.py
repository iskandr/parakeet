"""Matrix multiplication four ways: operator, adverb, comprehension, loops."""

import numpy as np

import parakeet
import parakeet as par


@par.jit
def matmult_operator(X, Y):
  return X @ Y


@par.jit
def matmult_allpairs(X, Y):
  return parakeet.allpairs(np.dot, X, Y.T)


@par.jit
def matmult_comprehension(X, Y):
  return np.array([[np.dot(x, y) for y in Y.T] for x in X])


@par.jit
def matmult_loops(X, Y):
  out = np.zeros((X.shape[0], Y.shape[1]))
  for i in range(X.shape[0]):
    for j in range(Y.shape[1]):
      for k in range(X.shape[1]):
        out[i, j] += X[i, k] * Y[k, j]
  return out


def main(n: int = 120):
  rng = np.random.default_rng(5)
  X, Y = rng.random((n, n + 3)), rng.random((n + 3, n - 2))
  for f in (matmult_operator, matmult_allpairs, matmult_comprehension, matmult_loops):
    assert np.allclose(f(X, Y), X @ Y), f.__name__
  return X @ Y


if __name__ == "__main__":
  print(main().shape)
