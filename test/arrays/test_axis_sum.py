import numpy as np
import parakeet
from parakeet.testing_helpers import assert_eq_arrays, run_local_tests

@parakeet.jit
def each_sum(X):
  return parakeet.each(parakeet.sum, X)

m = 2000
n = 1000
X = np.random.random((m,n))

def test_each_row_sum():
  assert_eq_arrays(np.sum(X, axis = 1), each_sum(X), "sum rows")

@parakeet.jit
def col_sum(X):
  return np.sum(X, axis = 0)

def test_each_col_sum():
  assert_eq_arrays(np.sum(X, axis = 0), col_sum(X), "sum cols")

if __name__ == '__main__':
  run_local_tests()
