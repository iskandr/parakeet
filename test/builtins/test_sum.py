import numpy as np 
from parakeet.testing_helpers import expect, run_local_tests


def test_sum_int():
  x = np.array([1,2,3])
  expect(sum, [x], sum(x))

def test_sum_float():
  x = np.array([1.0,2.0,3.0])
  expect(sum, [x], sum(x))

def sum_bool():
  x = np.array([True, False, True])
  expect(sum, [x], sum(x))

if __name__ == '__main__':
  run_local_tests()
