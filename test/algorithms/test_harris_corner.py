import numpy as np
import time 

import parakeet 
from parakeet.testing_helpers import expect_each, run_local_tests, eq


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


def harris(I):
  dx = diff_x(I)
  dy = diff_y(I)
  #
  #   At each point we build a matrix
  #   of derivative products
  #   M =
  #   | A = dx^2     C = dx * dy |
  #   | C = dy * dx  B = dy * dy |
  #
  #   and the score at that point is:
  #      det(M) - k*trace(M)^2
  #
  A = dx * dx
  B = dy * dy
  C = dx * dy
  tr = A + B
  det = A * B - C * C
  k = 0.05
  return det - k * tr * tr


def test_harris():
  expect_each(harris, harris, matrices)


  
if __name__ == '__main__':
  run_local_tests()
