import numpy as np
import time 
from parakeet import jit 
from testing_helpers import run_local_tests, expect, eq

int_mat = np.reshape(np.arange(9), (3,3))
float_mat = np.sqrt(int_mat)
bool_mat = int_mat % 2

matrices = [int_mat, float_mat, bool_mat]

@jit 
def mm(X,Y,Z):
  m,d = X.shape
  n = Y.shape[1]
    
  for i in range(m):
      for j in range(n):
          total = 0
          for k in range(d):
              total += X[i,k] * Y[k,j]
          Z[i,j] = total 
  return Z

def test_loop_matmult():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y)
      Z = np.zeros(res.shape, dtype = res.dtype)
      expect(mm, [X,Y,Z], res)

if __name__ == '__main__':
    run_local_tests()
