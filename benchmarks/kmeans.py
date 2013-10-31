import parakeet
import numpy as np 


def dist(x,y):
  return ((x-y)**2).sum()

def kmeans_comprehensions(X, k, niters = 10):
  C = X[:k, :]
  for _ in xrange(niters):
    A = np.array([np.argmin([dist(x,c) for c in C]) for x in X])
    C = np.array([np.mean(X[A == i, :], axis = 0) for i in xrange(k)])
  return C

def kmeans_loops(X, k, niters = 10):
  C = X[:k, :]
  n,ndims = X.shape
  A = np.zeros(n, dtype=int)
  for _ in xrange(niters):
    # assign data points to nearest centroid
    for i in xrange(n):
      x = X[i,:]
      min_dist = dist(x, C[0, :]) 
      min_idx = 0 
      for cidx in xrange(1,k):
        centroid = C[cidx,:]
        curr_dist = 0.0
        for xidx in xrange(ndims):
          curr_dist += (x[xidx] - centroid[xidx])**2
        if curr_dist < min_dist:
          min_dist = curr_dist
          min_idx = cidx
      A[i] = min_idx
    # recompute the clusters by averaging data points 
    # assigned to them 
    for cidx in xrange(k):
      # reset centroids
      for dim_idx in xrange(ndims):
        C[cidx, dim_idx] = 0
      # add each data point only to its assigned centroid
      cluster_count = 0
      for i in xrange(n):
        if A[i] == cidx:
          C[cidx, :] += X[i, :]
          cluster_count += 1
      C[cidx, :] /= cluster_count 
  return C      

n, d = 10**4, 50
X = np.random.randn(n,d)
k = 25

from compare_perf import compare_perf

compare_perf(kmeans_comprehensions, [X, k, 5],cpython=False)

compare_perf(kmeans_loops, [X, k, 5], cpython=True)
