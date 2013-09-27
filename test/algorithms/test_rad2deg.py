
from parakeet.testing_helpers import run_local_tests, expect, expect_eq

def rad2deg(rad):
  return (rad * 180) / 3.14159265359

import numpy as np 

def test_rad2deg_int32():
  rads = np.array([0,1]).astype('int32')
  degs = rad2deg(rads)
  expected = np.rad2deg(rads)
  expect_eq(degs, expected)

def test_rad2deg_bool():
  rads = np.array([False,True])
  degs = rad2deg(rads)
  expected = np.rad2deg(rads)
  expect_eq(degs, expected)

def test_rad2deg_float32():
  rads = np.array([0.1, 0.2]).astype('float32')
  degs = rad2deg(rads)
  expected = np.rad2deg(rads)
  expect_eq(degs, expected)


def test_rad2deg_float64():
  rads = np.array([0.1, 0.2]).astype('float64')
  degs = rad2deg(rads)
  expected = np.rad2deg(rads)
  expect_eq(degs, expected)

if __name__ == '__main__':
  run_local_tests()




