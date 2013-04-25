import numpy as np 
from decorators import jit 

@jit
def pmap1d(f, x, w = 3):
  n = x.shape[0]
  h = w / 2
  return [f(x[max(i-h, 0):min(i+h+1, n)]) for i in range(n)]


@jit  
def pmap2d(f, x, width = (3,3), step = (1,1)):
  """
  Patch-map where the function can accept both interior windows
  and smaller border windows
  """
  width_x, width_y = width 
  step_x, step_y = step 
  n_rows, n_cols = x.shape
  hx = width_x / 2
  hy = width_y / 2
  return [[f(x[max(i-hx, 0):min(i+hx+1, n_rows), max(j-hy, 0):min(j+hy+1, n_cols)]) 
           for j in np.arange(0, n_cols, step_x)] 
           for i in np.arange(0, n_rows, step_y)]
  
@jit  
def pmap2d_trim(f, x, width = (3,3), step = (1,1)):
  """
  Patch-map over interior windows, ignoring the border
  """
  width_x, width_y = width 
  step_x, step_y = step 
  n_rows, n_cols = x.shape
  hx = width_x / 2
  hy = width_y / 2
  return [[f(x[i-hx:i+hx+1, j-hy:j+hy+1]) 
           for j in np.arange(hx, n_cols-hx, step_x)] 
           for i in np.arange(hy, n_rows-hy, step_y)]



