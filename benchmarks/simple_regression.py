from parakeet import jit 

import timer  

def covariance(x,y):
  return ((x-x.mean()) * (y-y.mean())).mean()

def fit_simple_regression(x,y):
  slope = covariance(x,y) / covariance(x,x)
  offset = y.mean() - slope * x.mean() 
  return slope, offset

import numpy as np 

N = 10**7
x = np.random.randn(N).astype('float64')
slope = 903.29
offset = 102.1
y = slope * x + offset

jit(fit_simple_regression)(x,y)

timer.compare_perf(fit_simple_regression, (x,y))
