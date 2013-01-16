import numpy as np
import time

import testing_helpers

from parakeet import allpairs

x = 8000
y = 8000
k = 500
x2_array = np.arange(x*k, dtype = np.float).reshape(x,k) / float(x*k)
y2_array = np.arange(k*y, 2*k*y, dtype = np.float).reshape(y,k) / float(y*k)

def dot(x, y):
  return sum(x*y)

def adverb_matmult(X, Y):
  return allpairs(dot, X, Y)

def test_par_mm():
  start = time.time()
  rslt = adverb_matmult(x2_array, y2_array)
  comp_time = time.time() - start

  check = True
  if check:
    start = time.time()
    adverb_matmult(x2_array, y2_array)
    par_time = time.time() - start

    start = time.time()
    nprslt = np.dot(x2_array, y2_array.T)
    np_time = time.time() - start
    assert(testing_helpers.eq(rslt, nprslt)), \
        "Expected %s but got %s" % (nprslt, rslt)
    print "NumPy time:", np_time
    print "Parakeet without compilation:", par_time

  print "Parakeet time with compilation:", comp_time

if __name__ == '__main__':
  testing_helpers.run_local_tests()
