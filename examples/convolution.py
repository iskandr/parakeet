import numpy as np 
from parakeet import jit 
# Simple convolution of 3x3 patches from a given array x
# by a 3x3 array of filter weights
 
@jit
def conv_3x3_trim(x, weights):
  return np.array([[(x[i-1:i+2, j-1:j+2]*weights).sum() 
                    for j in xrange(1, x.shape[1] -2)]
                    for i in xrange(1, x.shape[0] -2)])
 

x = np.random.randn(50,50)
w = np.random.randn(3,3)

print "Input", x
print "Convolved output", conv_3x3_trim(x,w)


