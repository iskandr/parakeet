import numpy as np
import parakeet 

from parakeet.testing_helpers import expect, run_local_tests 

def test_ones_1d():
  expect(np.ones, [(10,)], np.ones(10))

def test_ones_2d():
  expect(np.ones, [(10,2)], np.ones((10,2)))

def test_ones_1d_int64():
  expect(np.ones, [(10,), np.int64], np.ones(10, dtype=np.int64))

def test_ones_2d_int64():
  expect(np.ones, [(10,2), np.int64], np.ones((10,2), dtype=np.int64))

def test_ones_1d_float64():
  expect(np.ones, [(10,), np.float64], np.ones(10, dtype=np.float64))

def test_ones_2d_float64():
  expect(np.ones, [(10,2), np.float64], np.ones((10,2), dtype=np.float64))

def test_ones_4d_float64():
  expect(np.ones, [(10,2,2,2), np.float64], np.ones((10,2,2,2), dtype=np.float64))

if __name__ == "__main__":
  run_local_tests()
