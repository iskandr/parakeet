
import numpy as np 
import parakeet 
from parakeet import jit, testing_helpers

@jit
def maxpool(array, pool_size):
  n, w, h = array.shape
  def _inner(args):
    img, x, y = args
    max_val = -1e12
    for i in xrange(0, pool_size):
      for j in xrange(0, pool_size):
        if x + i < w and y + j < h:
          max_val = max(array[img, x + i, y + j])
    return max_val
  return parakeet.imap(_inner, (n, w, h))

def test_maxpool():
  x = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])
  xs = np.array([x,x])
  expected_single =  np.array([[5, 6, 6],
                               [8, 9, 9],
                               [8, 9, 9]])
  expected = np.array([expected_single, expected_single])
  testing_helpers.expect(maxpool, [xs, 2], expected)

if __name__ == "__main__":
  testing_helpers.run_local_tests()
