import config
import lib_simple

from adverb_api import allpairs, each, reduce, scan, conv
from lib_simple import *
from prims import *
from run_function import run, specialize_and_compile
from decorators import jit 


def typed_repr(fn, args):
  _, typed, _, _ = specialize_and_compile(fn, args)
  return typed

def sum(x):
  return reduce(add, x, init = 0)

def prod(x):
  return reduce(multiply, x[1:], init=x[0])

def mean(x):
  return sum(x) / x.shape[0]

def cumsum(x):
  return scan(add, x)

def cumprod(x):
  return scan(multiply, x)

def diff(x, zero_fill=True):
  """
  TODO:
    - axis selection
    - preserve size by filling with zeros
    - allow n'th differences by recursion
  """

  return x[1:] - x[:-1]

@jit
def dot(x,y):
  return sum(x*y)

def clear_specializations():
  import closure_type
  for clos_t in closure_type._closure_type_cache.itervalues():
    clos_t.specializations.clear()

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
