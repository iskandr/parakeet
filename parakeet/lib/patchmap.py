import numpy as np 
from .. frontend import jit 
from adverbs import imap 

@jit
def pmap1(f, x, w = 3):
  n = x.shape[0]
  def local_apply(i):
    lower = __builtins__.max(i-w/2, 0)
    upper = __builtins__.min(i+w/2+1, n)
    elts = x[lower:upper]
    return f(elts)
  return imap(local_apply, n)

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
    lx = __builtins__.max(i-hx, 0)
    ux = __builtins__.min(i+hx+1, n_rows)
    ly = __builtins__.max(j-hy, 0)
    uy = __builtins__.min(j+hy+1, n_cols)
    return f(x[lx:ux, ly:uy])
    
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



