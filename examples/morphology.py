
import numpy as np 
import time 

from parakeet import jit 
import scipy.ndimage

def dilate_naive(x, window_size):
  m,n = x.shape
  k,l = window_size 
  hk, hl = k/2, l/2
  y = np.empty_like(x)
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

@jit 
def dilate_decompose_loops(x, window_size):
  m,n = x.shape
  k,l = window_size 
  hk, hl = k/2, l/2
  y = np.empty_like(x)
  z = np.empty_like(x)
  for i in xrange(m):
    for j in xrange(n):
      left_idx = max(0, i-hk)
      right_idx = min(m, i+hk+1) 
      currmax = x[left_idx, j]
      for ii in xrange(left_idx+1, right_idx):
        elt = x[ii, j]
        if elt > currmax:
          currmax = elt 
      y[i, j] = currmax 
  for i in xrange(m):
    for j in xrange(n):
      left_idx = max(0, j-hl)
      right_idx = min(n, j+hk+1)
      currmax = y[i,left_idx]  
      for jj in xrange(left_idx+1, right_idx):
        elt = y[i,jj]
        if elt > currmax:
            currmax = elt
      z[i,j] = currmax
  return z 


def dilate_1d_naive(x_strip, y_strip, window_size):
  """
  Given a 1-dimensional input and 1-dimensional output, 
  fill output with 1d dilation of input 
  """
  nelts = len(x_strip)
  half = window_size / 2 
  for idx in xrange(nelts):
    left_idx = max(idx-half,0)
    right_idx = min(idx+half+1, nelts)
    currmax = x_strip[left_idx]
    for j in xrange(left_idx+1, right_idx):
      elt = x_strip[j]
      if elt > currmax:
        currmax = elt
    y_strip[idx] = currmax 

@jit 
def dilate_decompose(x, window_size): 
  m,n = x.shape
  k,l = window_size
  y = np.empty_like(x)
  z = np.empty_like(x)

  for i in xrange(m):
    dilate_1d_naive(x[i,:], y[i,:], k)
  for j in xrange(n):
    dilate_1d_naive(y[:, j], z[:, j], l)
  return z

def dilate_1d_interior(x_strip, y_strip, window_size):
  nelts = len(x_strip)
  half = window_size / 2 
  
  interior_start = half+1
  interior_stop = max(nelts-half, interior_start)
  
  # left boundary
  for i in xrange(min(half+1, nelts)):
    left_idx = max(i-half,0)
    right_idx = min(i+half+1, nelts)
    currmax = x_strip[left_idx]
    for j in xrange(left_idx+1, right_idx):
      elt = x_strip[j]
      if elt > currmax:
        currmax = elt
    y_strip[i] = currmax 
    
  #interior 
  for i in xrange(interior_start, interior_stop):
    left_idx = i-half
    right_idx = i+half+1
    currmax = x_strip[left_idx]
    for j in xrange(left_idx+1, right_idx):
      elt = x_strip[j]
      if elt > currmax:
        currmax = elt
    y_strip[i] = currmax 
  
  # right boundary
  for i in xrange(interior_stop, nelts):
    left_idx = max(i-half, 0)
    right_idx = nelts
    currmax = x_strip[left_idx]
    for j in xrange(left_idx+1, right_idx):
      elt = x_strip[j]
      if elt > currmax:
        currmax = elt
    y_strip[i] = currmax 
  
@jit 
def dilate_decompose_interior(x, window_size): 
  m,n = x.shape
  k,l = window_size
  y = np.empty_like(x)
  z = np.empty_like(x)
  for row_idx in xrange(m):
    dilate_1d_interior(x[row_idx,:], y[row_idx,:], k)
  for col_idx in xrange(n):
    dilate_1d_interior(y[:, col_idx], z[:, col_idx], l)
  return z 


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
    
window_size = (11,11)
width, height = 1024,768
image = np.random.randint(0, 150,  (width, height))


with timer('scipy'):
  scipy_result = scipy.ndimage.grey_dilation(image, window_size, mode='nearest')

"""
with timer('cpython-naive'):
  naive_result = dilate_naive(image, window_size)
assert np.allclose(naive_result, scipy_result)
"""

dilate_naive = jit(dilate_naive)

def run(fn, name, imshow=False):
  print 
  with timer(name + '-compile'):
    fn(image[:1, :1], window_size)
  with timer(name):
    result = fn(image, window_size)
  
  if imshow:
    import pylab
    pylab.imshow(result)
    pylab.figure()
    pylab.imshow(scipy_result)
    pylab.show()
  assert np.allclose(result, scipy_result)
  
run(dilate_naive, 'parakeet-naive')
run(dilate_decompose_loops, 'decompose-loops')
run(dilate_decompose, 'decompose-slices', imshow=False)
run(dilate_decompose_interior, 'decompose-interior')


