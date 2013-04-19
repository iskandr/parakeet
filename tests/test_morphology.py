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


def load_img(path  = '../data/rjp_small.jpg', gray=True):
  x = pylab.imread(path)
  if len(x.shape) > 2 and gray:
    x =  x[:, :, 1] 
  x = x.astype('float') / 256.0
  return x


def test_erode():
  x = load_img(gray=False)
  
  print x.shape
  def filter(img):
    print img.shape
    return dilate(erode(img, (100,2)), (5,50))
  r = filter(x[:,:, 0])
  g = filter(x[:,:,1])
  b = filter(x[:,:,2])
  y = np.dstack([r,g,b])
  pylab.imshow(x, origin='lower')
  pylab.figure()
  pylab.imshow(y, origin='lower')
  pylab.show()
  assert (x.min() <= y).all()
 
 
if __name__ == '__main__':
  run_local_tests() 
