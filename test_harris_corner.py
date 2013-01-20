import numpy as np
import time 

import parakeet 

from testing_helpers import expect_each, run_local_tests

size = (5,5)
float_mat = np.random.uniform(0,1,size=size)
bool_mat = float_mat > 0.5
int_mat = np.random.random_integers(0,255,size=size)

matrices = [float_mat, bool_mat, int_mat]

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
"""
def parallel_harris(I):
  x_indices = np.arange(I.shape[0])[1:-1]
  y_indices = np.arange(I.shape[1])[1:-1]
   
  def scalar_computation(i,j):
    dx = I[i+1,j] - I[i,j]
    dy = I[i,j+1] - I[i,j]
    A = dx * dx
    B = dy * dy
    C = dx * dy
    tr = A + B
    det = A * B - C * C
    k = 0.05
    return det - k * tr * tr
  return parakeet.allpairs(scalar_computation, x_indices, y_indices)
"""
"""
def test_harris():
  expect_each(harris, harris, matrices)
"""
def test_harris_timing():
  x = np.random.randn(1500, 1500)
  
  np_start = time.time()
  np_rslt = harris(x)
  np_time = time.time() - np_start

  par_start = time.time()
  parakeet.run(harris,x)
  par_time = time.time() - par_start

  par_start_no_comp = time.time()
  parakeet.run(harris,x)
  par_time_no_comp = time.time() - par_start_no_comp

  print "Parakeet time: %.3f" % par_time
  print "Parakeet w/out compilation: %.3f" % par_time_no_comp
  print "Python time: %.3f" % np_time
  assert par_time_no_comp / np_time < 5, \
    "Parakeet too slow (%.1fX slowdown)" % (par_time_no_comp / np_time)
  
  """
  start = time.time()
  parallel_result = parallel_harris(x)
  parallel_time = time.time() - start
  print "Parallel time: %.3f" % parallel_time
  rmse = np.sqrt(np.mean( (parallel_result - np_rslt) **2))
  assert rmse < 0.001, rmse 
  """
  
if __name__ == '__main__':
  run_local_tests()
