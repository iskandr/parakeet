import numpy as np

from parakeet import jit 
from parakeet.testing_helpers import run_local_tests, expect

int64_mat = np.reshape(np.arange(6), (2,3)).astype('int64')
int32_mat = int64_mat.astype('int32')

float64_mat = int64_mat.astype('float64')
float32_mat = int64_mat.astype('float32') 
bool_mat = int64_mat % 2

matrices = [int64_mat, int32_mat, float64_mat, float32_mat, bool_mat]

def matmult_comprehensions(X,Y):
  return np.array([[np.dot(x,y) for y in Y.T] for x in X])

def test_matmult_int64_int32():
  expect(matmult_comprehensions, [int64_mat, int32_mat.T], np.dot(int64_mat, int32_mat.T))

def test_matmult_int32_int32():
  expect(matmult_comprehensions, [int32_mat, int32_mat.T], np.dot(int32_mat, int32_mat.T))

def test_matmult_float64_float32():
  expect(matmult_comprehensions, [float64_mat, float32_mat.T], np.dot(float64_mat, float32_mat.T))

def test_matmult_float32_float32():
  expect(matmult_comprehensions, [float32_mat, float32_mat.T], np.dot(float32_mat, float32_mat.T))

def test_matmult_float32_int32():
  expect(matmult_comprehensions, [float32_mat, int32_mat.T], np.dot(float32_mat, int32_mat.T))

def test_matmult_float64_bool():
  expect(matmult_comprehensions, [float64_mat, bool_mat.T], np.dot(float64_mat, bool_mat.T))

def test_matmult_bool_bool():
  expect(matmult_comprehensions, [bool_mat, bool_mat.T], np.dot(bool_mat, bool_mat.T))

def test_matmult_bool_int32():
  expect(matmult_comprehensions, [bool_mat, int32_mat.T], np.dot(bool_mat, int32_mat.T))
      
if __name__ == "__main__":
  run_local_tests()
