
import numpy as np 
import time 

from parakeet import jit 
import scipy.ndimage

def dilate_naive(x, window_size):
  m,n = x.shape
  k,l = window_size 
  hk, hl = k/2, l/2
  y = np.zeros_like(x)
  for i in xrange(m):
    for j in xrange(n):
      currmax = x[i,j]
      for ii in xrange(max(0, i-hk), min(m, i+hk+1)):
        for jj in xrange(max(0, j-hl), min(n, j+hk+1)):
          elt = x[ii,jj]
          if elt > currmax:
            currmax = elt
      y[i,j] = currmax
  return y  

def dilate_decomposition(x, window_size):
  m,n = x.shape
  k,l = window_size 
  hk, hl = k/2, l/2
  y = np.zeros_like(x)
  for i in xrange(m):
    for j in xrange(n):
      currmax = x[i,j]
      for ii in xrange(max(0, i-hk), min(m, i+hk+1)):
        elt = x[ii, j]
        if elt > currmax:
          currmax = elt 
      y[i,j] = currmax 
  for i in xrange(m):
    for j in xrange(n):
      currmax = x[i,j]  
      for jj in xrange(max(0, j-hl), min(n, j+hk+1)):
        elt = x[i,jj]
        if elt > currmax:
            currmax = elt
      y[i,j] = currmax
  return y  

  

width, height = 10,10
image = np.random.randint(0, 50,  (width, height))

class timer(object):
  def __init__(self, name = None):
    self.name = name 
    self.start_t = time.time()
    
  def __enter__(self):
    self.start_t = time.time()
  
  def elapsed(self):
    return time.time() - self.start_t
  
  def __exit__(self,*exit_args):
    t = self.elapsed()
    if self.name is None:
      print "Elasped time %0.4f" % t 
    else:
      print "%s : elapsed time %0.4f" % (self.name, t) 

def eq(x,y):
  return hasattr(x, 'shape') and hasattr(y, 'shape') and x.shape == y.shape and (x==y).all()
    
window_size = (3,3)
with timer('cpython-naive'):
  naive_result = dilate_naive(image, window_size)

with timer('parakeet-naive-parse'):
  naive_dilate = jit(dilate_naive)

with timer('parakeet-naive-compile'):
  naive_dilate(image[:1,:1], window_size) 
  
with timer('parakeet-naive-run'):
  naive_jit_result = dilate_naive(image, window_size)

print "jit result difference", np.linalg.norm(naive_result - naive_jit_result ) 

with timer('scipy'):
  scipy_result = scipy.ndimage.grey_dilation(image, window_size, mode='nearest')

print "scipy result difference", np.linalg.norm(naive_result - scipy_result)

with timer('parakeet-decompose')