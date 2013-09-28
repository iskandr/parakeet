import numpy as np

import parakeet 
from parakeet import testing_helpers

def erode(x, shape):
  return parakeet.pmap2(min, x, shape)

x = np.array([[1,2,3],[4,5,6]])

def test_erode_identity_shape():
  testing_helpers.expect(erode, [x, (1,1)], x)

def test_erode():
  shape = (3,3)
  expected = np.array([[1, 1, 2], [1,1,2]])
  testing_helpers.expect(erode, [x, shape], expected)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
