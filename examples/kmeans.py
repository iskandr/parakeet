import parakeet
import numpy as np 
from timer import timer 


def kmeans_comprehensions(X, k, niters = 10):
  def dist(x,y):
    return ((x-y)**2).sum()
  C = X[:k, :]
  for _ in xrange(niters):
    A = np.array([np.argmin([dist(x,c) for c in C]) for x in X])
    C = np.array([np.mean(X[A == i, :], axis = 0) for i in xrange(k)])
  return C

def kmeans_loops(X, k, niters = 10):
  def dist(x,y):
    return ((x-y)**2).sum()
  C = X[:k, :]
  n,ndims = X.shape
  dists = np.zeros(k, X.dtype)
  A = np.zeros(n, dtype=int)
  for _ in xrange(niters):
    # assign data points to nearest centroid
    for i in xrange(n):
      x = X[i,:]
      min_dist = dist(x, C[0, :]) 
      min_idx = 0 
      for cidx in xrange(1,k):
        centroid = C[cidx,:]
        curr_dist = dist(x,centroid)
        if curr_dist < min_dist:
          min_dist = curr_dist
          min_idx = cidx
  for cidx in xrange(k):
    # reset centroids
    C[cidx, :] = 0
    # add each data point only to its assigned centroid
    for i in xrange(n):
      if A[i] == cidx:
        C[cidx, :] += X[i, :]
      
fast_kmeans = parakeet.jit(kmeans)  

n, d = 10**4, 100
X = np.random.randn(n,d)
k = 5

with timer('parakeet first run'):
  fast_kmeans(X, k)

with timer('parakeet second run'):
  fast_kmeans(X, k)

with timer('python'):
  kmeans(X, k)
