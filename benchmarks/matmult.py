import parakeet
import numpy as np 

def matmult_high_level(X,Y):
  return np.array([[np.dot(x,y) for y in Y.T] for x in X])

def matmult_loops(X,Y,Z):
  m, d = X.shape
  n = Y.shape[1]
  for i in xrange(m):
    for j in xrange(n):
      total = X[i,0] * Y[0,j] 
      for k in xrange(1,d):
        total += X[i,k] * Y[k,j]
      Z[i,j] = total 
  return Z

n, d = 1200, 500
m = 1200
dtype = 'float32'
X = np.random.randn(m,d).astype(dtype)
Y = np.random.randn(d,n).astype(dtype)
Z = np.zeros((m,n)).astype(dtype)
from compare_perf import compare_perf

compare_perf(matmult_high_level, [X,Y],cpython=False, numba=False,extra = {'numpy':np.dot}, suppress_output = False)
compare_perf(matmult_loops, [X, Y, Z], cpython=False)

