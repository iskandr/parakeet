import numpy as np 
import parakeet 
from testing_helpers import expect, run_local_tests, eq, expect_allpairs 

int_mat = np.reshape(np.arange(9), (3,3))
float_mat = np.sqrt(int_mat)
bool_mat = int_mat % 2


def dot(x,y):
  return sum(x*y)

def test_dot():
  vecs = [m[0] for m in matrices]
  expect_allpairs(dot, np.dot, vecs)

def adverb_matmult(X,Y):
  return parakeet.allpairs(dot, X, Y)

matrices = [int_mat, float_mat, bool_mat]

def transposed_np_dot(x,y):
  return np.dot(x, y.T)

def test_adverb_matmult():
  expect_allpairs(adverb_matmult, transposed_np_dot, matrices)

if __name__ == '__main__':
  run_local_tests()