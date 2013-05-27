import numpy as np
import parakeet 

from testing_helpers import expect, run_local_tests 


int_vec = np.array([1,2,3])
float_vec = np.array([1.0,2.0,3.0])
bool_vec = np.array([True,False])

int_mat = np.array([int_vec, int_vec])
float_mat = np.array([float_vec, float_vec])
bool_mat = np.array([bool_vec, bool_vec])

def test_ones():
  expect(parakeet.ones, [(10,)], np.ones(10))
  expect(parakeet.ones, [(10,2)], np.ones((10,2)))  
  expect(parakeet.ones, [(10,), np.int64], np.ones(10, dtype=np.int64))
  expect(parakeet.ones, [(10,2), np.int64], np.ones((10,2), dtype=np.int64))
  expect(parakeet.ones, [(10,), np.float64], np.ones(10, dtype=np.float64))
  expect(parakeet.ones, [(10,2), np.float64], np.ones((10,2), dtype=np.float64))
  expect(parakeet.ones, [(10,2,2,2), np.float64], np.ones((10,2,2,2), dtype=np.float64))

def test_ones_like():
  expect(parakeet.ones_like, [int_vec], np.ones_like(int_vec))
  expect(parakeet.ones_like, [int_mat], np.ones_like(int_mat))
  expect(parakeet.ones_like, [float_vec], np.ones_like(float_vec))
  expect(parakeet.ones_like, [float_mat], np.ones_like(float_mat))
  expect(parakeet.ones_like, [bool_vec], np.ones_like(bool_vec))
  expect(parakeet.ones_like, [bool_mat], np.ones_like(bool_mat))
  expect(parakeet.ones_like, [bool_vec, np.dtype('float')], np.ones_like(bool_vec, dtype = np.dtype('float')))
  expect(parakeet.ones_like, [bool_mat, np.dtype('float')], np.ones_like(bool_mat, dtype = np.dtype('float')))


def test_zeros():
  expect(parakeet.zeros, [(10,)], np.zeros(10))
  expect(parakeet.zeros, [(10,2)], np.zeros((10,2)))  
  expect(parakeet.zeros, [(10,), np.int64], np.zeros(10, dtype=np.int64))
  expect(parakeet.zeros, [(10,2), np.int64], np.zeros((10,2), dtype=np.int64))
  expect(parakeet.zeros, [(10,), np.float64], np.zeros(10, dtype=np.float64))
  expect(parakeet.zeros, [(10,2), np.float64], np.zeros((10,2), dtype=np.float64))
  expect(parakeet.zeros, [(10,2,2,2), np.float64], np.zeros((10,2,2,2), dtype=np.float64))

def test_zeros_like():
  expect(parakeet.zeros_like, [int_vec], np.zeros_like(int_vec))
  expect(parakeet.zeros_like, [int_mat], np.zeros_like(int_mat))
  expect(parakeet.zeros_like, [float_vec], np.zeros_like(float_vec))
  expect(parakeet.zeros_like, [float_mat], np.zeros_like(float_mat))
  expect(parakeet.zeros_like, [bool_vec], np.zeros_like(bool_vec))
  expect(parakeet.zeros_like, [bool_mat], np.zeros_like(bool_mat))
  expect(parakeet.zeros_like, [bool_vec, np.dtype('float')], np.zeros_like(bool_vec, dtype = np.dtype('float')))
  expect(parakeet.zeros_like, [bool_mat, np.dtype('float')], np.zeros_like(bool_mat, dtype = np.dtype('float')))


def test_empty():
  s = (20, 20, 3)
  x = parakeet.empty(s)
  assert x.shape == s
  
def test_empty_int():
  s = (2,2,2,2)
  x = parakeet.empty(s, dtype=np.uint8)
  assert x.shape == s
  assert x.dtype == np.uint8


if __name__ == '__main__':
  run_local_tests()
