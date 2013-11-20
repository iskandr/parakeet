import numpy as np 
from parakeet.testing_helpers import expect, run_local_tests

# Simple convolution of 3x3 patches from a given array x
# by a 3x3 array of filter weights
 
def conv_3x3_trim(x, weights):
  return np.array([[(x[i-1:i+2, j-1:j+2]*weights).sum() 
                    for j in xrange(1, x.shape[1] -1)]
                    for i in xrange(1, x.shape[0] -1)])
 
x = np.random.randn(4,4)
w = np.random.randn(3,3)
"""
def test_conv_float32():
    x_f32 = x.astype('float32')
    w_f32 = w.astype('float32')
    expect(conv_3x3_trim, [x_f32, w_f32], conv_3x3_trim(x_f32,w_f32))

def test_conv_int16():
    x_i16 = x.astype('int16')
    w_i16 = w.astype('int16')
    expect(conv_3x3_trim, [x_i16, w_i16], conv_3x3_trim(x_i16,w_i16))

def test_conv_bool():
    xb = x > 0
    wb = w > 0
    expect(conv_3x3_trim, [xb, wb], conv_3x3_trim(xb,wb))
"""
def conv_3x3_trim_loops(image, weights):
  result = np.zeros_like(image[1:-1, 1:-1])
  m,n = image.shape 
  for i in xrange(1,m-1):
    for j in xrange(1,n-1):
      for ii in xrange(3): 
        for jj in xrange(3):
          result[i-1,j-1] += image[i-ii+1, j-jj+1] * weights[ii, jj] 
  return result

def test_conv_loops_float32():
    x_f32 = x.astype('float32')
    w_f32 = w.astype('float32')
   
    expect(conv_3x3_trim_loops, [x_f32, w_f32], conv_3x3_trim_loops(x_f32,w_f32))

def test_conv_loops_int16():
    x_i16 = x.astype('int16')
    w_i16 = w.astype('int16')
    expect(conv_3x3_trim_loops, [x_i16, w_i16], conv_3x3_trim_loops(x_i16,w_i16))

def test_conv_loops_bool():
    xb = x > 0
    wb = w > 0
    expect(conv_3x3_trim_loops, [xb, wb], conv_3x3_trim_loops(xb,wb))

if __name__ == "__main__":
    run_local_tests()
