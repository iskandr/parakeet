import numpy as np
import time

from parakeet import allpairs, multiply, run, sum
from testing_helpers import expect, expect_allpairs, run_local_tests

bool_vec = np.array([True, False, True])
int_vec = np.array([1,2,3])
float_vec = np.array([10.0, 20.0, 30.0])

vectors = [bool_vec, int_vec, float_vec]

def loop_outer_prod(x,y,z):
  nx = x.shape[0]
  ny = y.shape[0]
  i = 0
  while i < nx:
    j = 0
    while j < ny:
      z[i,j] = x[i] * y[j]
      j = j + 1
    i = i + 1
  return z

def test_loop_outer_prod():
  for v1 in vectors:
    for v2 in vectors:
      res = np.outer(v1, v2)
      v3 = np.zeros_like(res)
      expect(loop_outer_prod, [v1, v2, v3], res)

def adverb_outer_prod(x,y):
  return allpairs(multiply, x, y)

def test_adverb_outer_prod():
  expect_allpairs(adverb_outer_prod, np.multiply.outer, vectors)

int_mat = np.reshape(np.arange(9), (3,3))
float_mat = np.sqrt(int_mat)
bool_mat = int_mat % 2

def loop_matmult(X, Y, Z):
  n_rows = X.shape[0]
  row_length = X.shape[1]
  n_cols = Y.shape[1]
  i = 0
  while i < n_rows:
    j = 0
    while j < n_cols:
      total = X[i, 0] * Y[0, j]
      k = 1
      while k < row_length:
        total = total + X[i, k] * Y[k, j]
        k = k + 1
      Z[i, j] = total
      j = j + 1
    i = i + 1
  return Z

def test_loop_matmult():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y)
      Z = np.zeros(res.shape, dtype = res.dtype)
      expect(loop_matmult, [X,Y,Z], res)

def test_loop_matmult_timing():
  X = np.random.randn(100,100).astype('float32')
  Y = np.random.randn(100,100).astype('float32')
  
  start = time.time()
  _ = np.dot(X,Y.T)
  np_interval = time.time() - start

  res = np.zeros_like(Y)
  start = time.time()
  run(loop_matmult, X, Y, res)
  parakeet_interval = time.time() - start

  start = time.time()
  run(loop_matmult, X, Y, res)
  parakeet_interval_no_comp = time.time() - start
  print "Loop matrix multiplication timings"
  print "Numpy time:", np_interval
  print "Parakeet time:", parakeet_interval
  print "Parakeet time (w/out compilation):", parakeet_interval_no_comp
  assert float(parakeet_interval_no_comp) / np_interval < 100

def dot(x,y):
  return sum(x*y)

def adverb_matmult(X,Y):
  return allpairs(dot, X, Y)

matrices = [int_mat, float_mat, bool_mat]

def transposed_np_dot(x,y):
  return np.dot(x, y.T)

def test_adverb_matmult():
  expect_allpairs(adverb_matmult, transposed_np_dot, matrices)

if __name__ == '__main__':
  run_local_tests()
