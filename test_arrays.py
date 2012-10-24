import numpy as np 
from parakeet import expect
def create_const(x):
  return [x,x,x,x]

def test_create_const():
  expect(create_const, [1],  np.array([1,1,1,1]))
  expect(create_const, [1.0], np.array([1.0, 1.0, 1.0, 1.0]))
  expect(create_const, [True], np.array([True, True, True, True]))

def set_val(arr,idx,val):
  arr[idx] = val
  return arr 

def test_set_val():
  expect(set_val, [np.array([0,1,2,3]), 1, 100], np.array([0,100,2,3]))  
  expect(set_val, [np.array([0.0,1,2,3]), 1, 100], np.array([0.0, 100.0, 2.0, 3.0]))

if __name__ == '__main__':
  import testing_helpers
  testing_helpers.run_local_tests()

