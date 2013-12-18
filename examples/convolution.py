import numpy as np 
from parakeet import jit 
# Simple convolution of 3x3 patches from a given array x
# by a 3x3 array of filter weights
 
@jit
def conv_3x3_trim(x, weights):
  def compute_pixel(i,j):
    total = 0
    for ii in xrange(3):
        for jj in xrange(3):
          total += weights[ii,jj] * x[i-ii+1, j-jj+1]
    return total 

  return np.array([[compute_pixel(i,j)
                    for j in xrange(1, x.shape[1] -2)]
                    for i in xrange(1, x.shape[0] -2)])
 

x = np.random.randn(50,50)
w = np.random.randn(3,3)

print "Input", x
print "Convolved output", conv_3x3_trim(x,w)


