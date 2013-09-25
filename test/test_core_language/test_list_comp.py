import numpy as np 
from parakeet.testing_helpers import expect, run_local_tests


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
  x = np.array([1.0,2.0,3.0])
  y = np.array([10,20])
  res = np.array(outer_prod(x,y))
  expect(outer_prod, [x,y], res)
  
def triple_nesting(x):
  return [[[zi + yi + xi for zi in x] for yi in x] for xi in x]

def test_triple_nesting():
  x = np.array([1,2,3])
  expect(triple_nesting, [x], np.array(triple_nesting(x)))

def repeat_elts(x):
  return [x[i:i+2] for i in range(len(x)-1)]

def test_repeat_elts():
  x = np.array([0,1,2,3,4,5])
  y = np.array([[0,1], [1,2], [2,3], [3,4], [4,5]])
  expect(repeat_elts, [x], y)


if __name__ == '__main__':
  run_local_tests()
  
