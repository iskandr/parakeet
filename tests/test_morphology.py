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
  return parakeet.pmap2d(parakeet.min_, X, window_size)

#def dilate(X, window_size = (3,3)):
#  return parakeet.pmap2d(parakeet.max, X, window_size)


def load_img(path  = '../data/rjp_small.jpg'):
  x = pylab.imread(path)
  if len(x.shape) > 2:
    x = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2]) / 3 
  x = x.astype('float') / x.max()
  return x


def test_erode():
  x = load_img()
  print x.shape
  y = erode(x)[:, :, 0]
  print y.shape
  import pylab
  pylab.imshow(x, cmap='gray')
  pylab.figure()
  pylab.imshow(y, cmap='gray')
  pylab.show()
  assert (x.min() <= y).all()
 
 
if __name__ == '__main__':
  run_local_tests() 
