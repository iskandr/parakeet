import numpy as np
import pylab 
import scipy
import scipy.ndimage 
import time 

import parakeet

from parakeet import jit, pmap2 
from testing_helpers import eq, run_local_tests



def erode(X, window_size = (3,3)):
  return pmap2(min, X, window_size)
 
def dilate(X, window_size = (3,3)):
  return pmap2(max, X, window_size)


def load_img(path  = 'data/rjp_small.png', gray=True):
  try:
    x = pylab.imread(path)
  except:
    x = pylab.imread('../' + path)
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


plot = False 

def test_erode():
  x = load_img(gray=False)
  size = (10,10)
  r,g,b = x[:,:,0],x[:,:,1],x[:,:,2]
  

  def filter(img):
    print "---"
    par_start_compile_t = time.time()
    res_par = erode(img[:1, :1], size)
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
      pylab.show()
    assert res_par.shape == res_sci.shape
    assert (res_par == res_sci).all(), \
      "# different elements: %d / %d" % ((res_par != res_sci).sum(), res_par.size)
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
