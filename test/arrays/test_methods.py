import numpy as np
from parakeet import testing_helpers, jit 

values = [np.array(0), np.array(1.0), np.array([True]), 
          np.array([1,2,3]), np.array([1.0, 2.0, 3.0]), np.array([True, False]),
          np.array([[1,2,3],[10,20,30]]), np.array([[1.0,2.0,3.0],[10.0,20.0, 30.0]]), 
          np.array([[True], [False]])
         ]

def run(fn, method_name):
  fn = jit(fn)
  for v in values:
    method = getattr(v, method_name)
    expected = method()
    result = fn(v)
    assert np.allclose(expected, result), "For test input %s, expected %s but got %s" % (v, expected, result)

def test_min():
  def call_min(x):
    return x.min()
  run(call_min, 'min')


def test_max():
  def call_max(x):
    return x.max()
  run(call_max, 'max')


def test_any():
  def call_any(x):
    return x.any()
  run(call_any, 'any')

def test_all():
  def call_all(x):
    return x.all()
  run(call_all, 'all')

def test_copy():
  def call_copy(x):
    return x.copy()
  run(call_copy, 'copy')
  x = np.array([1,2,3])
  y = jit(call_copy)(x)
  x[0] = 10
  assert y[0] == 1
  
def test_ravel():
  def call_ravel(x):
    return x.ravel()
  run(call_ravel, 'ravel')

def test_flatten():
  def call_flatten(x):
    return x.flatten()
  run(call_flatten, 'flatten')

if __name__ == '__main__':
  testing_helpers.run_local_tests() 
