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
  
def outer_prod(x, y):
  return [[xi * yi for xi in x] for yi in y]

def test_outer_prod():
  x = [1.0,2.0,3.0]
  y = [10,20]
  res = np.array(outer_prod(x,y))
  expect(outer_prod, [x,y], res)
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()
  
