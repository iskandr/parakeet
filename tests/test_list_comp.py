import numpy as np 
import testing_helpers 

from testing_helpers import expect

x = np.array([1,2,3,4,5])

def identity_comprehension(x):
  return [xi for xi in x]

def test_identity_comprehension():
  expect(identity_comprehension, [x], x)


def sqr_elts(x):
  return [xi**2 for xi in x]

def test_sqr_elts():
  expect(sqr_elts, [x], x**2)
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()
  
