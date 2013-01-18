import scipy.spatial
import numpy as np
import time

import adverb_api

from parakeet import allpairs
from testing_helpers import run_local_tests, eq

def sqr_dist(x,y):
  return sum((x-y) ** 2)

def cdist(X,Y):
  return allpairs(sqr_dist,X,Y)

def test_sqr_dist():
  x = 5000
  y = 1000
  k = 5000
  #X = np.random.randn(5000, 5000)
  #Y = np.random.randn(1000, 5000)
  X = np.arange(x*k, dtype = np.float).reshape(x,k) / (x*k)
  Y = np.arange(y*k, dtype = np.float).reshape(y,k) / (y*k)

  start_time = time.time()
  par_rslt = cdist(X, Y)
  par_time = time.time() - start_time

  start_time = time.time()
  _ = cdist(X, Y)
  no_comp_time = time.time() - start_time

  #start_time = time.time()
  #np_rslt = scipy.spatial.distance.cdist(X,Y, 'sqeuclidean')
  #np_time = time.time() - start_time

  print "Parallel runtime:", adverb_api.par_runtime
  print "Parakeet No Compilation Time:", no_comp_time
  print "Parakeet Time:", par_time
  #print "NumPy time:", np_time

  #assert eq(np_rslt, par_rslt), \
  #    "Expected %s but got back %s" % (np_rslt, par_rslt)

if __name__ == '__main__':
  run_local_tests()
