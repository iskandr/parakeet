import numpy as np
import parakeet 

from parakeet.testing_helpers import expect, run_local_tests 


int_vec = np.array([1,2,3])
float_vec = np.array([1.0,2.0,3.0])
bool_vec = np.array([True,False])

int_mat = np.array([int_vec, int_vec])
float_mat = np.array([float_vec, float_vec])
bool_mat = np.array([bool_vec, bool_vec])

#
# ONES
#

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

#
# ONES_LIKE
#

def test_ones_like():
  expect(np.ones_like, [int_vec], np.ones_like(int_vec))

def test_ones_like_intmat():
  expect(np.ones_like, [int_mat], np.ones_like(int_mat))

def test_ones_like_floatvec():
  expect(np.ones_like, [float_vec], np.ones_like(float_vec))

def test_ones_like_floatmat():
  expect(np.ones_like, [float_mat], np.ones_like(float_mat))

def test_ones_like_boolvec():
  expect(np.ones_like, [bool_vec], np.ones_like(bool_vec))

def test_ones_like_boolmat():
  expect(np.ones_like, [bool_mat], np.ones_like(bool_mat))

def test_ones_like_boolvec_to_float():
  expect(np.ones_like, [bool_vec, np.dtype('float')], np.ones_like(bool_vec, dtype = np.dtype('float')))

def test_ones_like_boolmat_to_float():
  expect(np.ones_like, [bool_mat, np.dtype('float')], np.ones_like(bool_mat, dtype = np.dtype('float')))


#
# ZEROS
#

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

#
# ZEROS_LIKE
#
def test_zeros_like_intvec():
  expect(np.zeros_like, [int_vec], np.zeros_like(int_vec))

def test_zeros_like_intmat():
  expect(np.zeros_like, [int_mat], np.zeros_like(int_mat))

def test_zeros_like_floatvec():
  expect(np.zeros_like, [float_vec], np.zeros_like(float_vec))

def test_zeros_like_floatmat():
  expect(np.zeros_like, [float_mat], np.zeros_like(float_mat))

def test_zeros_like_boolvec():
  expect(np.zeros_like, [bool_vec], np.zeros_like(bool_vec))

def test_zeros_like_boolmat():
  expect(np.zeros_like, [bool_mat], np.zeros_like(bool_mat))

def test_zeros_like_boolvec_to_float():
  expect(np.zeros_like, [bool_vec, np.dtype('float')], np.zeros_like(bool_vec, dtype = np.dtype('float')))

def test_zeros_like_boolmat_to_float():
  expect(np.zeros_like, [bool_mat, np.dtype('float')], np.zeros_like(bool_mat, dtype = np.dtype('float')))

#
# EMPTY
#

def test_empty():
  s = (20, 20, 3)
  x = np.empty(s)
  assert x.shape == s
  
def test_empty_int():
  s = (2,2,2,2)
  x = np.empty(s, dtype=np.uint8)
  assert x.shape == s
  assert x.dtype == np.uint8


if __name__ == '__main__':
  run_local_tests()
