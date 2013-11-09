from parakeet import jit, config, c_backend 
 


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



from compare_perf import compare_perf 
compare_perf(fit_simple_regression, (x,y))
