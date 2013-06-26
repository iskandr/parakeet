import numpy as np 

# Simple convolution of 3x3 patches from a given array x
# by a 3x3 array of filter weights
 
def conv_3x3_trim(x, weights):
  result = np.zeros_like(x)
  for i in xrange(1,x.shape[0]-2):
    for j in xrange(1,x.shape[1]-2):
      window = x[i-1:i+2, j-1:j+2]
      result[i, j] = (window*weights).sum()
  return result





from timer import compare_perf

x = np.random.randn(500,500)
w = np.random.randn(3,3)

compare_perf(conv_3x3_trim, [x,w])

