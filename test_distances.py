import scipy.spatial
import numpy as np
import time 

from parakeet import allpairs
from testing_helpers import expect, run_local_tests, eq

 

def sqr_dist(x,y):
  return sum( (x-y) ** 2)

def cdist(X,Y):
  return allpairs(sqr_dist,X,Y)

def test_sqr_dist():
  X = np.random.randn(10, 5)
  Y = np.random.randn(7, 5)
  expect(cdist, [X,Y],  scipy.spatial.distance.cdist(X,Y, 'sqeuclidean'))

def test_sqrt_dist_performance():
  print "All pairs distances"
  print "--------------------"
        
  for m in (1000,1024):
    for n in (32, 100):
      for d in (20, 1000,):
        print "m = %d, n = %d, d = %d" % (m,n,d)
        X = np.random.randn(m, d)
        Y = np.random.randn(n, d)
        
        start = time.time()
        np_result = scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')
        np_time = time.time() - start
        
        start = time.time()
        parakeet_result = cdist(X,Y)
        parakeet_time = time.time() - start 
        
        start = time.time()
        parakeet_result = cdist(X,Y)
        parakeet_no_comp = time.time() - start
        

        print "Parakeet:", parakeet_time
        print "Parakeet (no compilation):", parakeet_no_comp
        print "Numpy:", np_time
        speedup =  float(np_time) / parakeet_no_comp
        print "Speedup: %.1f" % speedup  
        print      
        slowdown = 1 / speedup
        assert slowdown < 5, "Parakeet was too slow! (%dX slower)" % int(slowdown)    
        

  
  

if __name__ == '__main__':
  run_local_tests()