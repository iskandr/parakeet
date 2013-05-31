import numpy as np
import pylab 
import scipy
import scipy.ndimage 
import time 

import parakeet

from parakeet import jit, pmap2 
from testing_helpers import eq, run_local_tests


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


def load_img(path  = 'data/rjp_small.png', gray=True):
  try:
    x = pylab.imread(path)
  except:
    x = pylab.imread('../' + path)
  if len(x.shape) > 2 and gray:
    x =  x[:, :, 2]
  if len(x.shape) > 2 and x.shape[2] == 4:
    x = x[:,:,:3]
  if x.max() > 1: 
    x = x.astype('float') / 257.0
  return x


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

  """
  x = load_img(gray=False)
  def filter(img):
    return dilate(img, (10,10))
  r,g,b = x[:,:,0],x[:,:,1],x[:,:,2]
  rd = filter(r)
  
  assert rd.shape == r.shape 
  gd = filter(g)
  bd = filter(b)
  if plot:
    pylab.imshow(x)
    pylab.figure()
    y = np.dstack([rd, gd, bd])
    pylab.imshow(y)
    pylab.figure()
    pylab.imshow((y-x)**3)
    pylab.show()
  assert (r <= rd).all(), np.sum(r > rd)
  assert (g <= gd).all(), np.sum(g > gd)
  assert (b <= bd).all(), np.sum(b > bd)
  """

"""
def test_erode():
  x = load_img(gray=False)
  size = (10,10)
  r,g,b = x[:,:,0],x[:,:,1],x[:,:,2]
  

  def filter(img):
    par_start_compile_t = time.time()
    print "---"
    res_par = erode(img, size)
    par_end_compile_t = time.time()
    print "Parakeet compile time: %0.3f" % (par_end_compile_t - par_start_compile_t)
    
    print "---"
    par_start_t = time.time()
    res_par = erode(img, size)
    par_end_t = time.time()
    print "Parakeet time: %0.3f" % (par_end_t - par_start_t)
    sci_start_t = time.time()
    res_sci = scipy.ndimage.grey_erosion(img, size, mode = 'nearest')
    sci_end_t = time.time()
    print "SciPy time: %0.3f" % (sci_end_t - sci_start_t)
    
    
    if plot:
      pylab.imshow(res_par)
      pylab.figure()
      pylab.imshow(res_sci)
      pylab.figure()
      pylab.imshow(res_sci - res_par)
      pylab.show()
    assert res_par.shape == res_sci.shape
    assert (res_par == res_sci).all(), \
      "# different elements: %d / %d" % ((res_par != res_sci).sum(), res_par.size)
    return res_par
  filter(r)
  filter(g)
  filter(b)
  
"""
def morph_open(x, erode_shape, dilate_shape = None):
  if dilate_shape is None: 
    dilate_shape = erode_shape
  return dilate(erode(x, erode_shape), dilate_shape)


def morph_close(x, dilate_shape, erode_shape = None):
  if erode_shape is None: 
    erode_shape = dilate_shape
  return erode(dilate(x, dilate_shape), erode_shape)


plot_rgb = False 

def test_residual():
  x = load_img(gray=False)
  s1 = (5,20)
  s2 = (17,3)
  def filter(img):
    return erode(dilate(img, (3,3)), (1,1))#  + erode(img, s2), s2) **3
    
  r = filter(x[:,:,0])
  g = filter(x[:,:,1])
  b = filter(x[:,:,2])
  y = np.dstack([r,g,b])
  if plot_rgb:
    pylab.imshow(x)
    pylab.figure()
    pylab.imshow(y)
    pylab.show()

if __name__ == '__main__':
  run_local_tests() 
