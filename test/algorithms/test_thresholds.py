import numpy as np 
import time 


import parakeet
from parakeet import testing_helpers 

def count_thresh_orig(values, thresh):
  n = 0
  for elt in values:
    n += elt < thresh
  return n

count_thresh = parakeet.jit(count_thresh_orig)

def np_thresh(values, thresh):
  return np.sum(values < thresh)

def par_thresh(values, thresh):
  return parakeet.sum(values < thresh)

def test_count_thresh():
  v = np.array([1.2, 1.4, 5.0, 2, 3])
  parakeet_result = count_thresh(v, 2.0)
  python_result = count_thresh_orig(v,2.0)
  testing_helpers.expect(count_thresh, [v, 2.0], count_thresh_orig(v, 2.0))


if __name__ == '__main__':
  testing_helpers.run_local_tests()
