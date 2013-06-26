import numpy as np

def dists(X,Y):
  return np.array([[np.sum( (x-y) ** 2) for x in X] for y in Y])

d = 100
X = np.random.randn(1000,d)
Y = np.random.randn(200,d)

from timer import compare_perf
compare_perf(dists, [X,Y])


