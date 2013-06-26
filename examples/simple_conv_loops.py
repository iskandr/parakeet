import numpy as np 
from timer import timer 

# Simple convolution of 3x3 patches from a given array x
# by a 3x3 array of filter weights
 
def conv_3x3_trim(x, weights):
  result = np.zeros_like(x)
  for i in xrange(1,x.shape[0]-2):
    for j in xrange(1,x.shape[1]-2):
      window = x[i-1:i+2, j-1:j+2]
      result[i, j] = (window*weights).sum()
  return result

x = np.random.randn(500,500)
w = np.random.randn(3,3)

import parakeet
parakeet_conv = parakeet.jit(conv_3x3_trim)


with timer('Parakeet #1'):
    parakeet_result = parakeet_conv(x,w)

with timer('Parakeet #2'):
    parakeet_result = parakeet_conv(x,w)

import numba 
numba_conv = numba.autojit(conv_3x3_trim)

with timer('Numba #1'):
    numba_result = numba_conv(x,w)

with timer('Numba #2'):
    numba_result = numba_conv(x,w)

with timer('Python'):
    python_result = conv_3x3_trim(x, w)

assert np.allclose(parakeet_result, python_result)
assert np.allclose(numba_result, python_result)
