import numpy as np 
import parakeet 
import testing_helpers

@parakeet.jit 
def simple_map(f, x):
  return parakeet.fill(x.shape, lambda idx: f(x[idx]) ) 

def test_simple_map():
  testing_helpers.expect(simple_map, [abs, [-1,-2]], np.array([1,2]))

@parakeet.jit 
def simple_allpairs(f, x, y):
  out_shape = x.shape[0], y.shape[0]
  return parakeet.fill(out_shape, lambda idx: f(x[idx[0]], y[idx[0]]))

