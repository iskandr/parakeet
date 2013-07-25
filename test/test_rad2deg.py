
from testing_helpers import run_local_tests, expect, expect_eq

def rad2deg(rad):
  return (rad * 180) / 3.14159265359

import numpy as np 
def test_rad2deg():
  rads = np.array([0,1,2])
  degs = rad2deg(rads)
  expected = np.rad2deg(rads)
  expect_eq(degs, expected)


if __name__ == '__main__':
  run_local_tests()




