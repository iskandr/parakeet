import numpy as np
import parakeet
from parakeet import testing_helpers

 

@parakeet.jit
def range1(n):
  return range(n)

def test_range1():
  n = 10
  numpy_result = np.arange(n)
  parakeet_result = range1(n)
  testing_helpers.assert_eq_arrays(numpy_result, parakeet_result, "range1")
  
@parakeet.jit
def range2(start, stop):
  return range(start, stop)

def test_range2():
  shape = (100,111)
  numpy_result = np.arange(*shape)
  parakeet_result = range2(*shape)
  testing_helpers.assert_eq_arrays(numpy_result, parakeet_result, "range2")
  
@parakeet.jit
def range3(start, stop, step):
  return range(start, stop, step)

def test_range3():
  shape = (10,20,2)
  numpy_result = np.arange(*shape)
  parakeet_result = range3(10,20,2)
  testing_helpers.assert_eq_arrays(numpy_result, parakeet_result, "range3")


"""
def test_big_step():
  shape = (10,20,100)
  numpy_result =  np.arange(*shape)
  parakeet_result = range3(*shape)
  
  assert_eq_arrays(numpy_result, parakeet_result,"range_big_step")
"""

if __name__ == '__main__':
  testing_helpers.run_local_tests()
