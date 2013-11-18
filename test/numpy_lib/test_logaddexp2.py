
import numpy as np
from parakeet.testing_helpers import expect, run_local_tests


int_vec = np.array([2,3])
int_mat = np.array([int_vec, int_vec])

bool_vec = np.array([True, True])
bool_mat = np.array([bool_vec, bool_vec])

float32_vec = np.array([1.0, 2.0], dtype='float32')
float32_mat = np.array([float32_vec, float32_vec])

float64_vec = np.array([1.0, 10.0])
float64_mat = np.array([float64_vec, float64_vec])

arrays = [int_vec, bool_vec, float32_vec, float64_mat]

def test_logaddexp2_bool_bool_vec():
  expect(np.logaddexp2, [bool_vec, bool_vec], np.logaddexp2(bool_vec,bool_vec))

def test_logaddexp2_bool_bool_mat():
  expect(np.logaddexp2, [bool_mat, bool_mat], np.logaddexp2(bool_mat,bool_mat))

def test_logaddexp2_int_bool_vec():
  expect(np.logaddexp2, [int_vec, bool_vec], np.logaddexp2(int_vec,bool_vec))

def test_logaddexp2_int_bool_mat():
  expect(np.logaddexp2, [int_mat, bool_mat], np.logaddexp2(int_mat,bool_mat))

def test_logaddexp2_int_int_vec():
  expect(np.logaddexp2, [int_vec, int_vec], np.logaddexp2(int_vec,int_vec))

def test_logaddexp2_int_int_mat():
  expect(np.logaddexp2, [int_mat, int_mat], np.logaddexp2(int_mat,int_mat))

def test_logaddexp2_int_float32_vec():
  expect(np.logaddexp2, [int_vec, float32_vec], np.logaddexp2(int_vec,float32_vec))

def test_logaddexp2_int_float32_mat():
  expect(np.logaddexp2, [int_mat, float32_mat], np.logaddexp2(int_mat,float32_mat))

def test_logaddexp2_int_float64_vec():
  expect(np.logaddexp2, [int_vec, float64_vec], np.logaddexp2(int_vec,float64_vec))

def test_logaddexp2_int_float64_mat():
  expect(np.logaddexp2, [int_mat, float64_mat], np.logaddexp2(int_mat,float64_mat))

def test_logaddexp2_float32_float64_vec():
  expect(np.logaddexp2, [float32_vec, float64_vec], np.logaddexp2(float32_vec,float64_vec))

def test_logaddexp2_float32_float64_mat():
  expect(np.logaddexp2, [float32_mat, float64_mat], np.logaddexp2(float32_mat,float64_mat))

if __name__ == "__main__":
  run_local_tests()

