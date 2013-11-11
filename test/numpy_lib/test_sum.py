import numpy as np
import parakeet
from parakeet.testing_helpers import expect, run_local_tests, expect_each


m = 5
n = 3
float_mat = np.random.random((m,n))
int_mat = float_mat.astype('int16')
bool_mat = int_mat > 0
matrices = [float_mat, int_mat, bool_mat]

def each_sum(X):
  return parakeet.each(np.sum, X)

def test_each_sum():
  expect_each(each_sum, lambda X: np.sum(X, axis=1), matrices)

def sum_cols(X):
  return np.sum(X, axis = 0)

def test_col_sum():
  expect_each(sum_cols, lambda X: np.sum(X, axis=0), matrices)

def sum_rows(X):
  return np.sum(X, axis = 1)

def test_sum_rows():
  expect_each(sum_rows, lambda X: np.sum(X, axis=1), matrices)

def test_sum_elts():
  expect_each(np.sum, np.sum, matrices)

if __name__ == '__main__':
  run_local_tests()
