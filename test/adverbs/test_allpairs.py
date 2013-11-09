import numpy as np 
import parakeet 
from parakeet.testing_helpers import expect, run_local_tests, eq, expect_allpairs 

int_mat = np.reshape(np.arange(9), (3,3))
float_mat = np.sqrt(int_mat)
bool_mat = int_mat % 2
matrices = [int_mat, float_mat, bool_mat]
vecs = [m[0] for m in matrices]

def dot(x,y):
  return sum(x*y)

def test_dot():
  expect_allpairs(dot, np.dot, vecs)

def adverb_matmult(X,Y):
  return parakeet.allpairs(dot, X, Y)


def test_adverb_matmult():
    expect_allpairs(adverb_matmult, lambda x, y: np.dot(x, y.T), matrices)

def allpairs_elt_diff(x,y):
    return parakeet.allpairs(lambda xi,yi: xi - yi, x, y)

def test_allpairs_elt_diff():
    def python_impl(x,y):
      nx = len(x)
      ny = len(y)
      result = np.zeros(shape = (nx,ny), dtype=(x[0]-y[0]).dtype)
      for i in xrange(nx):
          for j in xrange(ny):
              result[i,j] = x[i] - y[j]
      return result 
    expect_allpairs(allpairs_elt_diff, python_impl, vecs)


if __name__ == '__main__':
  run_local_tests()
