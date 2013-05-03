import numpy as np 
from parakeet import imap 
from decorators import jit 

@jit
def pmap1(f, x, w = 3):
  n = x.shape[0]
  h = w / 2
  return [f(x[max(i-h, 0):min(i+h+1, n)]) for i in range(n)]


@jit  
def pmap2(f, x, width = (3,3)):
  """
  Patch-map where the function can accept both interior windows
  and smaller border windows
  """
  width_x, width_y = width 
  n_rows, n_cols = x.shape
  hx = width_x / 2
  hy = width_y / 2
  def local_apply((i,j)):
    if i <= hx or i >= n_rows - hx or j < hy or j >= n_cols - hy:
      lx = max(i-hx, 0)
      ux = min(i+hx+1, n_rows)
      ly = max(j-hy, 0)
      uy = min(j+hy+1, n_cols)
      result = f(x[lx:ux, ly:uy])
    else:
      lx = i-hx
      ux = i+hx+1
      ly = j-hy
      uy = j+hy+1
      result = f(x[lx:ux, ly:uy])
    return result
    
  return imap(local_apply, x.shape)
    
@jit  
def pmap2_trim(f, x, width = (3,3), step = (1,1)):
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



