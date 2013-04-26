import numpy as np
import time 
from parakeet import jit 
from testing_helpers import run_local_tests, expect, eq

@jit 
def mm(X,Y,Z):
  m,d = X.shape
  n = Y.shape[1]  
  for i in range(m):
      for j in range(n):
          total = 0.0
          for k in range(d):
              total += X[i,k] * Y[k,j]
          Z[i,j] = total 

def test_mm():
    x = np.random.randn(10,10)
    y = x.copy()
    z = x.copy()
    mm(x,y,z)
    assert eq(z, np.dot(x,y))

if __name__ == '__main__':
    run_local_tests()
