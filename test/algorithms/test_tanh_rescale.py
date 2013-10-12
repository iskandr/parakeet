import numpy as np 
from parakeet import jit, testing_helpers


alpha = 0.5
beta = 0.3
x = np.array([1,2,3])
y = np.tanh(x) * alpha + beta

@jit
def numpy_expressions(x, alpha = 0.5, beta = 0.3):
  return np.tanh(x) * alpha + beta 

def test_numpy_expressions():
  res = numpy_expressions(x)
  assert np.allclose(res, y), "Expected %s but got %s" % (y, res)

@jit 
def loopy(x, alpha = 0.5, beta = 0.3):
  y = np.empty_like(x, dtype=float)
  for i in xrange(len(x)):
    y[i] = np.tanh(x[i]) * alpha + beta
  return y

def test_loopy():
  res = loopy(x)
  assert np.allclose(res, y), "Expected %s but got %s" % (y, res)

@jit
def comprehension(x, alpha = 0.5, beta = 0.3):
  return np.array([np.tanh(xi)*alpha + beta for xi in x])

def test_comprehensions():
  res = comprehension(x)
  assert np.allclose(res, y), "Expected %s but got %s" % (y, res)

if __name__ == "__main__":
  testing_helpers.run_local_tests()
  
