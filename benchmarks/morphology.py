
import platform 
impl = platform.python_implementation()

from timer import timer 

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

  
k = 7
width, height = 1024,768
if running_pypy:
  from random import randint
  image = np.array([[randint(0,256) for _ in xrange(width)] for _ in xrange(height)])
else:
  image = np.random.randint(0, 256,  (width, height)) / 256.0

def run(fn, name, imshow=False):
  print 
  print "---", name 
  for (prefix, wrapper) in [('parakeet-', jit), ('numba-', autojit)]:
    try:
      wrapped_fn = wrapper(fn)
      with timer(prefix + name + '-compile', True):
        wrapped_fn(image[:1, :1], k)
      with timer(prefix + name, False):
        result = wrapped_fn(image, k)
      if imshow:
        import pylab
        pylab.imshow(image)
        pylab.figure()
        pylab.imshow(result)
        pylab.figure()
        if scipy_result is not None:
          pylab.imshow(scipy_result)
          pylab.show()
      if not running_pypy and scipy_result is not None:
        assert allclose(result, scipy_result)

    except KeyboardInterrupt:
      raise   
    except:
      print "%s failed" % (prefix+name)
      import sys 
      print sys.exc_info()[1]

def dilate_naive(x, k):
  m,n = x.shape
  y = empty_like(x)
  for i in xrange(m):
    for j in xrange(n):
      currmax = x[i,j]
      for ii in xrange(max(0, i-k/2), min(m, i+k/2+1)):
        for jj in xrange(max(0, j-k/2), min(n, j+k/2+1)):
          elt = x[ii,jj]
          if elt > currmax:
            currmax = elt
      y[i,j] = currmax
  return y  

# Numba doesn't yet support min/max so try using inline if expressions
def dilate_naive_inline(x,k):
  m,n = x.shape
  y = empty_like(x)
  for i in xrange(m):
    start_i = i - k/2
    stop_i = i + k/2 + 1
    for j in xrange(n):
      currmax = x[i,j]
      start_j = j - k/2
      stop_j = j + k/2 + 1
      for ii in xrange(start_i if start_i >= 0 else 0, stop_i if stop_i <= m else m):
        for jj in xrange(start_j if start_j >= 0 else 0, stop_j if stop_j <=n else n):
          elt = x[ii,jj]
          if elt > currmax:
            currmax = elt
      y[i,j] = currmax
  return y  

 
def dilate_decompose_loops(x, k):
  m,n = x.shape
  y = empty_like(x)
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
  z = empty_like(x)
  for i in xrange(m):
    for j in xrange(n):
      left_idx = max(0, j-k/2)
      right_idx = min(n, j+k/2+1)
      currmax = y[i,left_idx]  
      for jj in xrange(left_idx+1, right_idx):
        elt = y[i,jj]
        if elt > currmax:
          currmax = elt
      z[i,j] = currmax
  return z 

def dilate_decompose_loops_inline(x, k):
  m,n = x.shape
  y = empty_like(x)
  for i in xrange(m):
    start_i = i-k/2
    stop_i = i+k/2+1 
    for j in xrange(n):
      left_idx = start_i if start_i >= 0 else 0
      right_idx = stop_i if stop_i <= m else m 
      currmax = x[left_idx, j]
      for ii in xrange(left_idx+1, right_idx):
        elt = x[ii, j]
        if elt > currmax:
          currmax = elt 
      y[i, j] = currmax 
  z = empty_like(x)
  for i in xrange(m):
    for j in xrange(n):
      start_j = j-k/2
      stop_j = j+k/2+1
      left_idx = start_j if start_j >= 0 else 0
      right_idx = stop_j if stop_j <= n else n
      currmax = y[i,left_idx]  
      for jj in xrange(left_idx+1, right_idx):
        elt = y[i,jj]
        if elt > currmax:
            currmax = elt
      z[i,j] = currmax
  return z 


def dilate_1d_naive(x_strip,  k):
  """
  Given a 1-dimensional input and 1-dimensional output, 
  fill output with 1d dilation of input 
  """
  nelts = len(x_strip)
  y_strip = empty_like(x_strip)
  half = k / 2 
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

def dilate_decompose(x, k): 
  m,n = x.shape
  y = np.array([dilate_1d_naive(x[row_idx, :], k) for row_idx in xrange(m)])
  return np.array([dilate_1d_naive(y[:, col_idx], k) for col_idx in xrange(n)]).T

def dilate_1d_interior(x_strip, k):
  
  nelts = len(x_strip)
  y_strip = empty_like(x_strip)
  half = k / 2 
  
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

def dilate_decompose_interior(x, k): 
  m,n = x.shape
  y = np.array([dilate_1d_interior(x[row_idx, :],k) for row_idx in xrange(m)])
  return np.array([dilate_1d_interior(y[:, col_idx],k) for col_idx in xrange(n)]).T
 
if __name__ == '__main__':
  if not running_pypy: 
    scipy_result = None 
    import scipy.ndimage
    with timer('scipy'):
      scipy_result = scipy.ndimage.grey_dilation(image, k, mode='nearest')
    from numba import autojit 
    from parakeet import jit 
    jit(dilate_naive)(image, k)

    run(dilate_naive, 'naive', imshow=False)
    run(dilate_naive_inline, 'naive-inline')
    run(dilate_decompose_loops, 'decompose-loops')    
    run(dilate_decompose_loops_inline, 'decompose-loops-inline')
    run(dilate_decompose, 'decompose-slices' )
    run(dilate_decompose_interior, 'decompose-interior')

  with timer('cpython-naive'):
    dilate_naive(image, k,)
  with timer('cpython-naive-inline'):
    dilate_naive_inline(image, k)
  with timer('cpython-decompose-loops'):
    dilate_decompose_loops(image, k)
  with timer('cpython-decompose-loops-inline'):
    dilate_decompose_loops_inline(image, k)
