
import platform 
impl = platform.python_implementation()

print "Running under", impl
running_pypy = impl == 'PyPy'


if running_pypy:
  import numpypy as np 
  # crazy that NumPyPy doesn't support this
  def allclose(x,y):
    if x.shape != y.shape:
      return False
    err = ((x-y)**2).mean()
    return err < 0.000001
  def empty_like(x):
    return np.empty(x.shape, dtype=x.dtype)
else:
  import numpy as np 
  allclose = np.allclose
  empty_like = np.empty_like 
import time 

if not running_pypy:
  from numba import autojit 
  from parakeet import jit 

def dilate_naive(x, window_size):
  m,n = x.shape
  k,l = window_size 
  y = empty_like(x)
  for i in xrange(m):
    for j in xrange(n):
      currmax = x[i,j]
      for ii in xrange(max(0, i-k/2), min(m, i+k/2+1)):
        for jj in xrange(max(0, j-l/2), min(n, j+l/2+1)):
          elt = x[ii,jj]
          if elt > currmax:
            currmax = elt
      y[i,j] = currmax
  return y  

if not running_pypy:
  dilate_naive_parakeet = jit(dilate_naive)
  dilate_naive_numba = autojit(dilate_naive)

def dilate_decompose_loops(x, window_size):
  m,n = x.shape
  k,l = window_size 
  y = empty_like(x)
  z = empty_like(x)
  for i in xrange(m):
    for j in xrange(n):
      left_idx = max(0, i-k/2)
      right_idx = min(m, i+k/2+1) 
      currmax = x[left_idx, j]
      for ii in xrange(left_idx+1, right_idx):
        elt = x[ii, j]
        if elt > currmax:
          currmax = elt 
      y[i, j] = currmax 
  for i in xrange(m):
    for j in xrange(n):
      left_idx = max(0, j-l/2)
      right_idx = min(n, j+l/2+1)
      currmax = y[i,left_idx]  
      for jj in xrange(left_idx+1, right_idx):
        elt = y[i,jj]
        if elt > currmax:
            currmax = elt
      z[i,j] = currmax
  return z 

if not running_pypy:
  dilate_decompose_loops_parakeet = jit(dilate_decompose_loops)
  dilate_decompose_loops_numba = autojit(dilate_decompose_loops)

def dilate_1d_naive(x_strip,  window_size):
  """
  Given a 1-dimensional input and 1-dimensional output, 
  fill output with 1d dilation of input 
  """
  nelts = len(x_strip)
  y_strip = empty_like(x_strip)
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
  return y_strip

def dilate_decompose(x, window_size): 
  m,n = x.shape
  k,l = window_size
  y = [dilate_1d_naive(x[row_idx, :], k) for row_idx in xrange(m)]
  z = [dilate_1d_naive(y[:, col_idx], l) for col_idx in xrange(n)]
  return np.array(z).T

if not running_pypy:
  dilate_decompose_parakeet = jit(dilate_decompose)
  dilate_decompose_numba = autojit(dilate_decompose)

def dilate_1d_interior(x_strip, window_size):
  
  nelts = len(x_strip)
  y_strip = empty_like(x_strip)
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
  return y_strip 

def dilate_decompose_interior(x, window_size): 
  m,n = x.shape
  k,l = window_size
  y = [dilate_1d_interior(x[row_idx, :],k) for row_idx in xrange(m)]
  z = [dilate_1d_interior(y[:, col_idx],l) for col_idx in xrange(n)]
  return np.array(z).T 

if not running_pypy:
  dilate_decompose_interior_parakeet = jit(dilate_decompose_interior)
  dilate_decompose_interior_numba = autojit(dilate_decompose_interior)


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

  
window_size = (7,7)
width, height = 1024,768
if running_pypy:
  from random import randint
  image = np.array([[randint(0,256) for _ in xrange(width)] for _ in xrange(height)])
else:
  image = np.random.randint(0, 256,  (width, height)) / 256.0
#print image.dtype 
#import pylab
#image  = pylab.imread('../data/rjp_small.png')[:,:,1]

def run(fn, name, imshow=False):
  print 
  try:
    with timer(name + '-compile'):
      fn(image[:1, :1], window_size)
    with timer(name):
      result = fn(image, window_size)
    if imshow:
      import pylab
      pylab.imshow(image)
      pylab.figure()
      pylab.imshow(result)
      pylab.figure()
      pylab.imshow(scipy_result)
      pylab.show()
    if not running_pypy:
      assert allclose(result, scipy_result)
  except KeyboardInterrupt:
    raise   
  except:
    print "%s failed" % name
    import sys 
    print sys.exc_info()[1]
 
if __name__ == '__main__':

  if not running_pypy: 
    import scipy.ndimage
    with timer('scipy'):
      scipy_result = scipy.ndimage.grey_dilation(image, window_size, mode='nearest')
    #import skimage.morphology
    #with timer('skimage'):
    #  selem = skimage.morphology.selem.rectangle(*window_size)
    #  skimage_result = skimage.morphology.dilation(image, selem)
    run(dilate_naive_parakeet, 'parakeet-naive', imshow=False)
    run(dilate_naive_numba, 'numba-naive')
    run(dilate_decompose_loops_parakeet, 'parakeet-decompose-loops')
    run(dilate_decompose_loops_numba, 'numba-decompose-loops')

    run(dilate_decompose_parakeet, 'parakeet-decompose-slices' )
    run(dilate_decompose_numba, 'numba-decompose-slices-numba')
    run(dilate_decompose_interior_parakeet, 'parakeet-decompose-interior')
    run(dilate_decompose_interior_numba, 'numba-decompose-interior')

  run(dilate_naive, 'cpython-naive')
  run(dilate_decompose_loops, 'cpython-decompose=loops')
