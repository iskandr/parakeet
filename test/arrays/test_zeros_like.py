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
  
if __name__ == "__main__":
  run_local_tests()
  
  