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

if __name__ == '__main__':
  run_local_tests()
