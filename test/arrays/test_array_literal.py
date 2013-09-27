import numpy as np

from parakeet.testing_helpers import expect, run_local_tests

def create_const(x):
  return [x,x,x,x]

def test_create_const():
  expect(create_const, [1],  np.array([1,1,1,1]))
  expect(create_const, [1.0], np.array([1.0, 1.0, 1.0, 1.0]))
  expect(create_const, [True], np.array([True, True, True, True]))


def test_nested_2d():
  expect(create_const, [np.array([1,2])] , np.array([[1,2],[1,2],[1,2],[1,2]]))
  expect(create_const, [np.array([1.0,2.0])] , np.array([[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0]]))
  expect(create_const, [np.array([True,False])] , 
         np.array([[True,False],[True,False],[True,False],[True,False]]))
  
if __name__ == '__main__':
  run_local_tests()
