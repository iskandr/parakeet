import numpy as np

from parakeet import scan, add
from parakeet.testing_helpers import run_local_tests, expect, expect_each

int_1d = np.arange(5)
float_1d = np.arange(5, dtype='float')
int_2d = np.array([int_1d, int_1d, int_1d])
float_2d = np.array([float_1d, float_1d, float_1d])

def running_sum(x):
  return scan(add, x, init = 0)

def test_scan_add_1d():
  expect_each(running_sum, np.cumsum, [int_1d, float_1d])

def loop_row_sums(x):
  n_rows = x.shape[0]
  y = np.zeros_like(x)
  y[0, :] = x[0, :]
  for i in xrange(1,n_rows):
    y[i, :] = y[i-1, :] + x[i, :]
  return y
"""
def test_scan_add_2d():
  expect_each(running_sum, loop_row_sums, [int_2d, float_2d])
"""
if __name__ == '__main__':
  run_local_tests()
