import parakeet
import numpy as np 
from timer import timer 

def dist(x,y):
  return ((x-y)**2).sum()

def kmeans(X, k, niters = 10):
  C = X[:k, :]
  for _ in xrange(niters):
    A = [np.argmin([dist(x,c) for c in C]) for x in X]
    C = [np.mean(X[A == i, :], axis = 0) for i in xrange(k)]
  return C

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
