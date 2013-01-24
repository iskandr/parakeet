import numpy as np
import time

import testing_helpers

from parakeet import allpairs, runtime 

x = 200
y = 10
k = 10

x2_array = np.arange(x*k, dtype = np.float).reshape(x,k) / (x*k)

y2_array = np.arange(k*y, 2*k*y, dtype = np.float).reshape(y,k) / (y*k)
y2T = y2_array #y2_array.copy(order='F')

def dot(x, y):
  return sum(x*y)

def adverb_matmult(X, Y):
  return allpairs(dot, X, Y)

def test_par_mm():
  start = time.time()
  rslt = adverb_matmult(x2_array, y2T)
  comp_time = time.time() - start

  start = time.time()
  adverb_matmult(x2_array, y2T)
  par_time = time.time() - start
  print "Parakeet without compilation:", par_time
  print "Parakeet time with compilation:", comp_time
  check = True
  if check:
    start = time.time()
    nprslt = np.dot(x2_array, y2T.T)
    np_time = time.time() - start
    print "NumPy time:", np_time
    assert(testing_helpers.eq(rslt, nprslt)), \
      "Expected %s but got %s" % (nprslt, rslt)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
