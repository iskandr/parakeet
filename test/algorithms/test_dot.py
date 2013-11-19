import numpy as np
from parakeet.testing_helpers import  expect_allpairs, run_local_tests

bool_vec = np.array([True, False, True, ])
int_vec = np.array([1,2,3,])
float_vec = np.array([10.0, 20.0, 30.0 ])

vectors = [bool_vec, int_vec, float_vec]

def loop_dot(x,y):
  n = x.shape[0]
  result = x[0] * y[0]
  i = 1
  while i < n:
    result += x[i] * y[i]
    i = i + 1
  return result

def test_loopdot():
  expect_allpairs(loop_dot, np.dot, vectors)

def dot(x,y):
  return sum(x*y)

def test_adverb_dot():
  expect_allpairs(dot, lambda x,y: np.sum(x*y), vectors)

if __name__ == '__main__':
  run_local_tests()
