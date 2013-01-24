import numpy as np
import scipy.spatial 
import time 

import parakeet 
import testing_helpers

def sqr_dist(x,y):
  return sum( (x-y) ** 2)

def apd(X,Y):
  return parakeet.allpairs(sqr_dist, X, Y)



def test_dists():
  m = 500 
  n = 200
  d = 10
  X = np.random.randn(m,d)
  Y = np.random.randn(n,d)

  start = time.time()
  _ = apd(X,Y)
  parakeet_time = time.time() - start 
  
  start = time.time()
  parakeet_dists = apd(X,Y)
  parakeet_no_comp_time = time.time() - start
  
  start = time.time()
  python_dists = scipy.spatial.distance.cdist(X,Y,'sqeuclidean')
  np_time = time.time() - start 
  
  print "Parakeet time", parakeet_time
  print "Parakeet (no compilation)", parakeet_no_comp_time
  print "NumPy time", np_time
  
  assert testing_helpers.eq(parakeet_dists, python_dists)    
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()
