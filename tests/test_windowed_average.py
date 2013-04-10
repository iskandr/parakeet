import parakeet 
from parakeet import jit 
import numpy as np
import testing_helpers
 
 
@jit
def avg2d(x):
  nelts = x.shape[0] * x.shape[1]
  return sum(sum(x)) / float(nelts)

def test_avg2d():
  x = np.random.randn(20,30)
  testing_helpers.eq(x.mean(), avg2d(x))

@jit
def winavg2d( x, wx = 3, wy = 3):
  return parakeet.winmap2d(avg2d, x, wx, wy)

"""
def test_winavg2d():
  x = np.random.randn(100,100)
  y = winavg2d(x)
  assert x.shape==y.shape
  assert x.max() >= y.max()
  assert x.min() <= y.min()
"""
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()
  