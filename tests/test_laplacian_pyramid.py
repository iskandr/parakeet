import numpy as np
import math
import pylab 
import scipy
import scipy.weave

def upsample(small, new_rows = None, new_cols = None):
  old_rows, old_cols = small.shape
  if new_rows is None:
    new_rows = old_rows * 2
  if old_cols is None:
    new_cols = old_cols * 2
  big = np.zeros(new_rows, new_cols)
  row_ratio = new_rows / float(old_rows)
  col_ratio = new_cols / float(old_colks)
  radius = 2 
  result = np.zeros(new_rows, new_cols)
  for i in xrange(new_rows):
    for j in xrange(new_cols):
      if i < radius or j < radius or (new_rows - i - 1) < radius or \
        (new_cols - j - 1) < radius:
        result[i,j] = 0  
      else:
        window = small[i-radius:i+radius, j-radius:j+radius]
        result[i,j] = np.sum(window*weights)
           


def conv(x, y, weights):
  return scipy.weave.inline("""
    int w = Nweights[0];
    int half_w = w / 2; 
    int n_rows = Nx[0];
    int n_cols = Nx[1];

    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        if (i > half_w && i < n_rows - half_w && j > half_w && j < n_cols - half_w) {
          y[i,j] = 0.0; 
          for (int ii = 0; ii < w; ++ii) {
            for (int jj = 0; jj < w ; ++jj) {
              y[i,j] += x[i + ii - half_w, j + jj - half_w] * weights[ii, jj];
            }
          }
        } 
      }
    }
    return_val = y;
  """, ('x', 'y', 'weights'))
      
  
def blur(x, radius = 2):
  n_rows, n_cols = x.shape
  window_width = radius*2 + 1
  sqr_dists = np.array([[(i-radius)**2 + (j-radius)**2 
                         for j in xrange(window_width)]
                         for i in xrange(window_width)])
  weights = np.exp(-sqr_dists)
  # normalize so weights add up to 1
  weights /= np.sum(weights)
  y = x.copy()
  conv(x,y,weights) 
  return y
  
def downsample(x):
  xb = blur(x)
  n_rows, n_cols = xb.shape
  result = xb[1::2, 1::2]
  print x.shape
  print xb.shape
  print result.shape
  return result

def test():
  x = pylab.imread('../data/bv.tiff')
  if len(x.shape) > 2:
    x = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2]) / 3 
  
  x = x.astype('float') / x.max()
  print 'orig', x.min(), x.max()
  pylab.imshow(x, cmap='gray')
  pylab.title('original')
  x_blur = blur(x)
  print 'blur', x_blur.min(), x_blur.max()
  pylab.figure()
  pylab.imshow(x_blur, cmap='gray')
  pylab.title('blur')
  x_small = downsample(x)
  print 'small', x_small.min(), x_small.max()
  pylab.figure()
  pylab.imshow(x_small, cmap='gray')
  pylab.title('small')
  x_small2 = downsample(x_small)
  pylab.figure()
  pylab.imshow(x_small2, cmap='gray')
  pylab.title('smaller')
  pylab.show()
  
  