import numpy as np
import math
import pylab 
import scipy
import scipy.weave
import parakeet
from testing_helpers import eq, run_local_tests

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
           

def conv(x, weights):
  def f(window):
    return sum(sum(window*weights))
  return parakeet.pmap2d_trim(f, x, weights.shape)


def load_img(path  = 'data/rjp_small.png'):
  try:
    x = pylab.imread(path)
  except:
    x = pylab.imread('../' + path)
  if len(x.shape) > 2:
    x = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2]) / 3 
  x = x[:100, :100] 
  x = x.astype('float') / x.max()
  return x

def blur(x, radius = 2):
  n_rows, n_cols = x.shape
  window_width = radius*2 + 1
  sqr_dists = np.array([[(i-radius)**2 + (j-radius)**2 
                         for j in xrange(window_width)]
                         for i in xrange(window_width)])
  weights = np.exp(-sqr_dists)
  # normalize so weights add up to 1
  weights /= np.sum(weights)
  y = conv(x, weights)
  return y

plot = False

def test_blur():
  x = load_img()
  radius = 2
  y = blur(x, radius=radius)
  if plot:
    import pylab
    pylab.imshow(x)
    pylab.figure()
    pylab.imshow(y)
    pylab.show()
  print x[24,42]
  print y[24-radius,42-radius]
  print y[70,70]
  assert abs(x[24,42] - y[24-radius,42-radius]) < abs(x[24,42] - y[70,70])

 
def downsample(x):
  xb = blur(x)
  n_rows, n_cols = xb.shape
  result = xb[1::2, 1::2]
  print x.shape
  print xb.shape
  print result.shape
  return result
 
if __name__ == '__main__':
  run_local_tests() 
