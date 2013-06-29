import numpy as np

def sqr_dists(X,Y):
  return np.array([[np.sum( (x-y) ** 2) for x in X] for y in Y])

def sqr_dists_loops(X,Y):
  m,n = X.shape[0], Y.shape[0]
  D = np.zeros((m,n), dtype=X.dtype)
  for i in xrange(m):
    for j in xrange(n):
      D[i,j] = np.sum( (X[i, :] -Y[j, :]) ** 2)
  return D

ndims = 10
nsamples = 10**4
nclusters = 50
X = np.random.randn(nsamples, ndims)
Y = np.random.randn(nclusters, ndims)




from timer import timer 

print 
print "Computing distances between %d and %d %s vectors of length %d" % \
  (nsamples, nclusters, X.dtype, ndims)

# 
# Parakeet 
# 

import parakeet

parakeet_dists = parakeet.jit(sqr_dists)

with timer('Parakeet (comprehensions) #1'):
  parakeet_dists(X,Y)

with timer('Parakeet (comprehensions) #2'):
  parakeet_dists(X,Y)

parakeet_dists_loops = parakeet.jit(sqr_dists_loops)

with timer('Parakeet (loops) #1'):
  parakeet_dists_loops(X,Y)

with timer('Parakeet (loops) #2'):
  parakeet_dists_loops(X,Y)



#
# Pure Python 
# 
from timer import timer

with timer('Python (comprehensions)'):
  sqr_dists(X,Y)

with timer('Python (loops)'):
  sqr_dists_loops(X,Y)



#
# Numba 
#

import numba

#
# Numba's @autojit just like Parakeet's @jit
#
numba_dists = numba.autojit(sqr_dists)

with timer('Numba (comprehensions) #1'):
  numba_dists(X,Y)

with timer('Numba (comprehensions) #2'):
  numba_dists(X,Y)

numba_dists_loops = numba.autojit(sqr_dists_loops)

with timer('Numba (loops) #1'):
  numba_dists_loops(X,Y)

with timer('Numba (loops) #2'):
  numba_dists_loops(X,Y)

