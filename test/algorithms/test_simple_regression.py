from parakeet.testing_helpers import expect, run_local_tests  


def covariance(x,y):
  return ((x-x.mean()) * (y-y.mean())).mean()

def fit_simple_regression(x,y):
  slope = covariance(x,y) / covariance(x,x)
  offset = y.mean() - slope * x.mean() 
  return slope, offset

import numpy as np 

def test_simple_regression():
  N = 10
  x = np.random.randn(N).astype('float64')
  slope = 903.29
  offset = 102.1
  y = slope * x + offset
  expect(fit_simple_regression, [x,y], (slope,offset))


if __name__ == '__main__':
  run_local_tests()



