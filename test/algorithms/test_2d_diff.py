
import numpy as np 
from parakeet.testing_helpers import expect_each, run_local_tests

size = (5,5)
float_mat = np.random.uniform(0,1,size=size)
int_mat = np.random.random_integers(0,255,size=size)

matrices = [float_mat, int_mat]

def diff_x(I):
  m = I.shape[0]
  return (I[1:, :] - I[:m-1, :])[:, 1:]

def test_diff_x():
  expect_each(diff_x, diff_x, matrices)

def diff_y(I):
  n = I.shape[1]
  return (I[:, 1:] - I[:, :n-1])[1:, :]

def test_diff_y():
  expect_each(diff_x, diff_x, matrices)

if __name__ == "__main__":
  run_local_tests()
