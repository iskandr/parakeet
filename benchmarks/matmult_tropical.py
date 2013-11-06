import parakeet
import numpy as np 


def dot(x,y):
    return np.min(x+y)

def matmult_high_level(X,Y):
  return np.array([[dot(x,y) for y in Y.T] for x in X])

def matmult_loops(X,Y,Z):
  m, d = X.shape
  n = Y.shape[1]
  for i in xrange(m):
    for j in xrange(n):
      total = X[i,0] + Y[0,j] 
      for k in xrange(1,d):
        total = min(total, X[i,k] + Y[k,j])
      Z[i,j] = total 
  return Z

n, d = 500, 500
m = 500
X = np.random.randn(m,d)
Y = np.random.randn(d,n)
Z = np.zeros((m,n))
from compare_perf import compare_perf

compare_perf(matmult_high_level, [X,Y], cpython=True, numba=False)
compare_perf(matmult_loops, [X, Y, Z], cpython=False)

