from parakeet import jit 
import numpy as np 


def dist(x,y):
  return ((x-y)**2).sum()

@jit 
def kmeans(X, k, niters = 10):
  C = X[:k, :]
  for _ in xrange(niters):
    A = np.array([np.argmin([dist(x,c) for c in C]) for x in X])
    C = np.array([np.mean(X[A == i, :], axis = 0) for i in xrange(k)])
  return C


n, d = 10**3, 100
X = np.random.randn(n,d)
k = 5

C = kmeans(X, k, niters=20) 
print "Clusters:", C
