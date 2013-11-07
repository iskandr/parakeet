import numpy as np 
from parakeet.testing_helpers import expect, run_local_tests

def test_range1():
  expect(range, [10], np.arange(10))

def test_range2():
  expect(range, [10, 20], np.arange(10,20))
  
def test_range3():
  expect(range, [20,45,3], np.arange(20,45,3))

if __name__ == '__main__':
  run_local_tests()
