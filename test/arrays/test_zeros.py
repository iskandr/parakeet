import numpy as np

from parakeet.testing_helpers import expect, run_local_tests 

def test_zeros_1d():
  expect(np.zeros, [(10,)], np.zeros(10))

def test_zeros_2d():
  expect(np.zeros, [(10,2)], np.zeros((10,2)))  

def test_zeros_1d_int64():
  expect(np.zeros, [(10,), np.int64], np.zeros(10, dtype=np.int64))

def test_zeros_2d_int64():
  expect(np.zeros, [(10,2), np.int64], np.zeros((10,2), dtype=np.int64))

def test_zeros_1d_float64():
  expect(np.zeros, [(10,), np.float64], np.zeros(10, dtype=np.float64))

def test_zeros_2d_float64():
  expect(np.zeros, [(10,2), np.float64], np.zeros((10,2), dtype=np.float64))

def test_zeros_4d_float64():
  expect(np.zeros, [(10,2,2,2), np.float64], np.zeros((10,2,2,2), dtype=np.float64))
  
  
if __name__ == "__main__":
  run_local_tests()
  
  