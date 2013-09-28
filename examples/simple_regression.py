from parakeet import jit 

def covariance(x,y):
  return ((x-x.mean()) * (y-y.mean())).mean()

@jit 
def fit_simple_regression(x,y):
  slope = covariance(x,y) / covariance(x,x)
  offset = y.mean() - slope * x.mean() 
  return slope, offset

import numpy as np 

N = 10**4
x = np.random.randn(N).astype('float64')
slope = 903.29
offset = 102.1
y = slope * x + offset

slope_estimate, offset_estimate = fit_simple_regression(x,y)

print "Expected slope =", slope, "offset =", offset
print "Parakeet slope =", slope_estimate, "offset =", offset_estimate
