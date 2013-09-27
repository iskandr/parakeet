import numpy as np 
import parakeet 
from parakeet import testing_helpers 

values = [np.array(0), np.array(0.0), np.array(True), 
          np.array([0]), 
          np.array([[0],[1]]), 
          np.array([[0.0], [1.0]]), 
          np.array([[True], [False]])
         ]

def run(fn, prop ):
  fn = parakeet.jit(fn)
  for v in values:
    expected = getattr(v, prop)
    result = fn(v)
    assert np.allclose(expected, result), "Expected %s but got %s" % (expected, result)
  

def test_prop_real():
  def get_real(x):
    return x.real 
  run(get_real, 'real')

#def test_prop_imag():
#  def get_imag(x):
#    return x.imag 
#  run(get_imag, 'imag')

def test_prop_ndim():
  def get_ndim(x):
    return x.ndim 
  run(get_ndim, 'ndim')

def test_prop_size():
  def get_size(x):
    return x.size
  run(get_size, 'size')

def test_prop_shape():
  def get_shape(x):
    return x.shape 
  run(get_shape, 'shape')

def test_prop_T():
  def get_T(x):
    return x.T
  run(get_T, 'T')

if __name__ == '__main__':
  testing_helpers.run_local_tests()
