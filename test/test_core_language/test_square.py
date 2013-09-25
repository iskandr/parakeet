from parakeet.testing_helpers import run_local_tests, expect_eq

def square(x):
  return np.square(x)

def test_square_scalars():
  expect_eq(square(1), 1)
  expect_eq(square(2), 4)
  expect_eq(square(-2), 4)
  expect_eq(square(True), True)
  expect_eq(square(False), False)
  expect_eq(square(-4.0), 16.0)

import numpy as np 
intvec = np.array([-3,-2,-1,0,1,2,3])
floatvec = intvec.astype('float')

def test_square_vectors():
  expect_eq(square(intvec), intvec*intvec)
  expect_eq(square(floatvec), floatvec*floatvec)

intmat = np.array([intvec, intvec])
floatmat = intmat.astype('float')

def test_square_matrices():
  expect_eq(square(intmat), intmat*intmat)
  expect_eq(square(floatmat), floatmat*floatmat)


if __name__ == '__main__':
  run_local_tests()
