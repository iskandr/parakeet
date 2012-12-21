import parakeet 
from testing_helpers import expect, expect_each, run_local_tests
import numpy as np

float_mat = np.random.uniform(0,1,size=(10,10))
bool_mat = float_mat > 0.5 
int_mat = np.random.random_integers(0,255,size=(10,10))

matrices = [float_mat, bool_mat, int_mat]

def diff_x(M):
  n = M.shape[0]
  return M[1:, :] - M[:n-1, :]

def test_diff_x():
  expect_each(diff_x, diff_x, matrices)
  
def diff_y(M):
  n = M.shape[1]
  return M[:, 1:] - M[:, :n-1]

def test_diff_y():
  expect_each(diff_x, diff_x, matrices)
  
def harris(M):
  dx = diff_x(M)[:, 1:]
  dy = diff_y(M)[1:, :]
  dx2 = dx*dx
  dy2 = dy*dy
  dxdy = dx * dy
  trace = dx2 + dy2 
  det = dx2 * dy2 - (dxdy*dxdy)
  k = 0.05
  return det -  k * trace * trace

def test_harris():
  expect_each(harris, harris, matrices)
   
if __name__ == '__main__':
  run_local_tests()  
  