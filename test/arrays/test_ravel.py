from parakeet import testing_helpers
import numpy as np 

def ravel_method(x):
  return x.ravel()

vec  = np.arange(12)

def run(x):
  testing_helpers.expect(ravel_method, [x], x.ravel())
  # testing_helpers.expect(np.ravel, [x], x.ravel())

def test_ravel_vec():
  run(vec)

def test_ravel_mat_row_major():
  matrix_row_major = np.reshape(vec, (3,4), order = "C")
  run(matrix_row_major)

def test_ravel_mat_col_major():
  matrix_col_major = np.reshape(vec, (3,4), order = "F")
  run(matrix_col_major)

def test_ravel_cube():
  cube = np.reshape(vec, (3,2,2))
  run(cube)

if __name__ == '__main__':
    testing_helpers.run_local_tests()
