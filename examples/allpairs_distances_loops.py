from timer import compare_with_numba 
import numpy as np

def dists(X,Y):
  result = np.zeros( (X.shape[0], Y.shape[0]), dtype=X.dtype)
  for i in xrange(X.shape[0]):
    for j in xrange(Y.shape[0]):
      result[i,j] = np.sum( (X[i,:] - Y[j,:]) ** 2)
  return result 

d = 100
X = np.random.randn(1000,d)
Y = np.random.randn(200,d)

compare_with_numba(dists, [X,Y])


