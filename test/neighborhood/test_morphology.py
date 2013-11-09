import numpy as np
import time 

import parakeet

from parakeet import jit, pmap2 
from parakeet.testing_helpers import eq, run_local_tests


def winmin(x):
  m,n = x.shape
  v = x[0,0]
  for i in range(m):
    for j in range(n):
      v2 = x[i,j]
      if v2 < v:
        v = v2
  return v

def winmax(x):
  m,n = x.shape
  v = x[0,0]
  for i in range(m):
    for j in range(n):
      v2 = x[i,j]
      if v2 > v:
        v = v2
  return v

def erode(X, window_size = (3,3)):
  return pmap2(winmin, X, window_size)
 
def dilate(X, window_size = (3,3)):
  return pmap2(winmax, X, window_size)



X = np.array([[0, 0,   0,   0,   0],
              [0, 0.1,  0.2, 0.3, 0],
              [0, 0.3,  0.4, 0.3, 0],
              [0, 0.2,  0.1, 0.5, 0],
              [0, 0,    0,   0,   0]])

# what we would expect after a 3x3 erosion 
X_erode = np.array([[0, 0, 0,   0, 0],
                    [0, 0, 0,   0, 0],
                    [0, 0, 0.1, 0, 0],
                    [0, 0, 0,   0, 0],
                    [0, 0, 0,   0, 0]])

X_dilate = np.array([[0.1, 0.2, 0.3, 0.3, 0.3],
                     [0.3, 0.4, 0.4, 0.4, 0.3],
                     [0.3, 0.4, 0.5, 0.5, 0.5],
                     [0.3, 0.4, 0.5, 0.5, 0.5],
                     [0.2, 0.2, 0.5, 0.5, 0.5]])

def test_erode():
  res = erode(X, (3,3))
  assert res.shape == X_erode.shape, "Expected shape %s but got %s" % (X_erode.shape, res.shape)
  assert (res == X_erode).all(), "Expected %s but got %s" % (X_erode, res)

def test_dilate():
  res = dilate(X, (3,3))
  assert res.shape == X_dilate.shape, "Expected shape %s but got %s" % (X_dilate.shape, res.shape)
  assert (res == X_dilate).all(), "Expected %s but got %s" % (X_dilate, res)



def dilate_1d_naive(x_strip, y_strip, window_size):
  """
  Given a 1-dimensional input and 1-dimensional output, 
  fill output with 1d dilation of input 
  """
  nelts = len(x_strip)
  half = window_size / 2 
  for i in xrange(nelts):
    left_idx = max(i-half,0)
    right_idx = min(i+half+1, nelts)
    currmax = x_strip[left_idx]
    for j in xrange(left_idx+1, right_idx):
      elt = x_strip[j]
      if elt > currmax:
        currmax = elt
    y_strip[i] = currmax 

@jit 
def dilate_decompose(x, window_size): 
  m,n = x.shape
  k,l = window_size
  y = np.empty_like(x)
  z = np.empty_like(x)
  for row_idx in xrange(m):
    dilate_1d_naive(x[row_idx,:], y[row_idx,:], k)
  for col_idx in xrange(n):
    dilate_1d_naive(y[:, col_idx], z[:, col_idx], l)
  return z

def test_dilate_decompose():
  res = dilate_decompose(X, (3,3))
  assert res.shape == X_dilate.shape, "Expected shape %s but got %s" % (X_dilate.shape, res.shape)
  assert (res == X_dilate).all(), "Original \n%s Expected dilation \n%s but got \n%s unequal elts \n%s" % \
    (X, X_dilate, res, X_dilate != res)

def morph_open(x, erode_shape, dilate_shape = None):
  if dilate_shape is None: 
    dilate_shape = erode_shape
  return dilate(erode(x, erode_shape), dilate_shape)


def morph_close(x, dilate_shape, erode_shape = None):
  if erode_shape is None: 
    erode_shape = dilate_shape
  return erode(dilate(x, dilate_shape), erode_shape)


plot_rgb = False 

if __name__ == '__main__':
  run_local_tests() 
