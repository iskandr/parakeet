import numpy as np
import parakeet 

from parakeet.testing_helpers import expect, run_local_tests 


def test_empty():
  s = (20, 20, 3)
  x = np.empty(s)
  assert x.shape == s
  
def test_empty_int():
  s = (2,2,2,2)
  x = np.empty(s, dtype=np.uint8)
  assert x.shape == s
  assert x.dtype == np.uint8


if __name__ == '__main__':
  run_local_tests()
