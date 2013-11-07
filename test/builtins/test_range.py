import numpy as np 
from parakeet import jit 
from parakeet.testing_helpers import expect, run_local_tests

@jit 
def range1(a):
  return range(a)

def test_range1():
  expect(range1, [10], np.arange(10))

@jit 
def range2(a,b):
  return range(a,b)

def test_range2():
  expect(range2, [10, 20], np.arange(10,20))

@jit
def range3(a,b,c):
  return range(a,b,c)
  
def test_range3():
  expect(range3, [20,45,3], np.arange(20,45,3))

if __name__ == '__main__':
  run_local_tests()
