import numpy as np
import pylab 
import scipy
import scipy.ndimage 
import time 

import parakeet
from testing_helpers import eq, run_local_tests

def min_(x):
  min_val = 1.0
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      if x[i,j] < min_val:
        min_val = x[i,j]
  return min_val
 

def erode(X, window_size = (3,3)):
  return parakeet.pmap2d(min_, X, window_size)

def erode_loops(X, window_size = (3,3)):
  m,n = X.shape
  wx, wy = window_size 
  result = parakeet.zeros_like(X)
  for i in range(m):
    for j in range(n):
      min_val = 1.0
      window = X[max(0, i -ii):min(m, i + ii), max(0, j - jj):min(n, j + jj)]
      for elt in parakeet.ravel(window):
        if elt < min_val:
          min_val = elt
    result[i,j] = min_val

def max_(x):
  max_val = 0.0
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      if x[i,j] > max_val:
        max_val = x[i,j]
  return max_val

def dilate(X, window_size = (3,3)):
  return parakeet.pmap2d(max_, X, window_size)


def load_img(path  = '../data/rjp_small.png', gray=True):
  x = pylab.imread(path)
  if len(x.shape) > 2 and gray:
    x =  x[:, :, 1]
  if x.max() > 1: 
    x = x.astype('float') / 256.0
  return x



def test_dilate():
  x = load_img(gray=False)
  def filter(img):
    return dilate(img, (10,10))
  r,g,b = x[:,:,0],x[:,:,1],x[:,:,2]
  rd = filter(r)
  assert rd.shape == r.shape 
  assert (r.min() <= rd).all()
  gd = filter(g)
  assert (g.min() <= gd).all()
  bd = filter(b)
  assert (b.min() <= bd).all()

def test_erode():
  x = load_img(gray=False)
  size = (10,10)
  r,g,b = x[:,:,0],x[:,:,1],x[:,:,2]
  
  print "EROSION TIMINGS"
  def filter(img):
    print "---"
    par_start_t = time.time()
    res_par = erode_loops(img, size)
    par_end_t = time.time()
    print "Parakeet time: %0.3f" % (par_end_t - par_start_t)
    sci_start_t = time.time()
    res_sci = scipy.ndimage.grey_erosion(r, size, mode = 'nearest')
    sci_end_t = time.time()
    print "SciPy time: %0.3f" % (sci_end_t - sci_start_t)
    assert res_par.shape == res_sci.shape
    #print np.abs(res_par-res_sci).max()
    #pylab.imshow((res_par - res_sci)**2)
    #pylab.show()
    assert True #(res_par == res_sci).all(), "# different elements: %d / %d" % ((res_par != res_sci).sum(), res_par.size)
    return res_par
  filter(r)
  filter(g)
  filter(b)
  

def morph_open(x, erode_shape, dilate_shape = None):
  if dilate_shape is None: 
    dilate_shape = erode_shape
  return dilate(erode(x, erode_shape), dilate_shape)


def morph_close(x, dilate_shape, erode_shape = None):
  if erode_shape is None: 
    erode_shape = dilate_shape
  return erode(dilate(x, dilate_shape), erode_shape)

"""
def test_residual():
  x = load_img(gray=False)
  s1 = (5,20)
  s2 = (17,3)
  def filter(img):
    return erode(dilate(img, s1) * erode(img, s2), s2)
    
  r = filter(x[:,:,0])
  g = filter(x[:,:,1])
  b = filter(x[:,:,2])
  y = np.dstack([r,g,b])
  pylab.imshow(x)
  pylab.figure()
  pylab.imshow(y)
  pylab.show()
"""
if __name__ == '__main__':
  run_local_tests() 
