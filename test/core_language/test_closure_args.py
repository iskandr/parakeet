from parakeet.testing_helpers import expect, run_local_tests
import numpy as np 

global_int = 5 
global_tuple = (5,3.0)
global_array = np.array([1,2,3])

def use_globals(x):
  return x + global_int + global_array + global_tuple[1]

def test_use_globals():
  expect(use_globals, [2], use_globals(2))

def use_locals(x):
  local_int = global_int
  local_array = global_array 
  local_tuple = global_tuple
  local_int2 = 3
  local_tuple2 = (5.0,9.0)
  local_array2 = np.array([4,5,6])
  def nested(y):
    return x + y + local_int + local_array + local_tuple[1] + local_int2 + local_tuple2[1] + local_array2
  return nested(x+1)

def test_use_locals():
  expect(use_locals, [True], use_locals(True))

if __name__ == "__main__":
  run_local_tests()
