from timer import compare_with_numba 
import numpy as np

def dists(X,Y):
  return np.array([[sum( (x-y) ** 2) for x in X] for y in Y])

d = 100
X = np.random.randn(1000,d)
Y = np.random.randn(200,d)

compare_with_numba(dists, [X,Y])


