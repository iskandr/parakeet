import numpy as np
import math
import pylab 
import scipy
import scipy.weave
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
  def filter(img):
    return erode(img, (10,10))
  r,g,b = x[:,:,0],x[:,:,1],x[:,:,2]
  rd = filter(r)
  assert rd.shape == r.shape 
  assert (r.max() >= rd).all()
  gd = filter(g)
  assert (g.max() >= gd).all()
  bd = filter(b)
  assert (b.max() >= bd).all()

@parakeet.jit 
def morph_open(x, erode_shape, dilate_shape = None):
  if dilate_shape is None: 
    dilate_shape = erode_shape
  return dilate(erode(x, erode_shape), dilate_shape)


def test_open():
  x = load_img(gray=False)
  
  def filter(img):
    return morph_open(img, (100,2), (5,50))
  r = filter(x[:,:,0])
  g = filter(x[:,:,1])
  b = filter(x[:,:,2])
  y = np.dstack([r,g,b])
  pylab.imshow(x)
  pylab.figure()
  pylab.imshow(y)
  pylab.show()
  
def morph_close(x, dilate_shape, erode_shape = None):
  if erode_shape is None: 
    erode_shape = dilate_shape
  return erode(dilate(x, dilate_shape), erode_shape)

if __name__ == '__main__':
  run_local_tests() 
