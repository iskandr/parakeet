import numpy as np
import scipy.spatial 
import time 

import parakeet 
from parakeet import testing_helpers

def sqr_dist(x,y):
  return sum( (x-y) ** 2)

def allpairs_dist_adverb(X,Y):
  return parakeet.allpairs(sqr_dist, X, Y)

def allpairs_dist_comprehensions_external(X,Y):
  return np.array([[sqr_dist(x,y) for y in Y] for x in X])

def allpairs_dist_comprehensions_internal(X,Y):
  def local_sqr_dist(x,y):
    return np.sum( (x-y)**2 )
  return np.array([[local_sqr_dist(x,y) for y in Y] for x in X])

m = 10 
n = 3
d = 5 
X = np.random.randn(m,d)
Y = np.random.randn(n,d)
python_dists = scipy.spatial.distance.cdist(X,Y,'sqeuclidean')

def test_dists_adverb():
  testing_helpers.expect(allpairs_dist_adverb, [X,Y], python_dists)

def test_dists_comprehensions_external():
  testing_helpers.expect(allpairs_dist_comprehensions_external, [X,Y], python_dists)
  
def test_dists_comprehensions_internal():
  testing_helpers.expect(allpairs_dist_comprehensions_external, [X,Y], python_dists)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
