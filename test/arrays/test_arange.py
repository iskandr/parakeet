
import numpy as np

from parakeet import jit, config 
from parakeet.testing_helpers import expect, run_local_tests 



@jit 
def arange1(a):
  return np.arange(a)

@jit  
def arange2(a,b):
  return np.arange(a,b)

@jit  
def arange3(a,b,c):
  return np.arange(a,b,c)

def test_range1_int():
  a = 5
  expect(arange1, [a], np.arange(a))

def test_range1_float():
  a = 3.5
  expect(arange1, [a], np.arange(a))
  
def test_range2_int():
  a = 2
  b = 6
  expect(arange2, [a,b], np.arange(a,b))

def test_range2_float():
  a = 3.5
  b = 7.1
  expect(arange2, [a,b], np.arange(a,b))
  
def test_range3_int():
  a = 2
  b = 10
  c = 3
  expect(arange3, [a,b,c], np.arange(a,b,c))

def test_range3_float():
  a = 3.5
  b = 9.1
  c = 1.7
  expect(arange3, [a,b,c], np.arange(a,b,c))

def test_range3_int_reverse():
  a = 10
  b = 2
  c = -3
  expect(arange3, [a,b,c], np.arange(a,b,c))

def test_range3_float_reverse():
  a = 9.1
  b = 3.5
  c = -1.7
  expect(arange3, [a,b,c], np.arange(a,b,c))  
    


if __name__ == '__main__':
  run_local_tests()
