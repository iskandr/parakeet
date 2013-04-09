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
def winmap2d(f, x, rows = 3, cols = 3):
  n_rows, n_cols = x.shape
  def window_fn(i,j):
    li = max(i-rows, 0)
    ui = min(i+rows, n_rows)
    lj = max(j-cols, 0)
    uj = min(j+cols, n_cols)
    return f(x[li:ui, lj:uj])
  return [[window_fn(i,j) 
           for j in np.arange(n_cols)] 
           for i in np.arange(n_rows)] 
  
@jit  
def winmap3d(f, x, wx = 3, wy = 3):
  n_rows, n_cols = x.shape
  def window_fn(i,j):
    li = max(i-wx, 0)
    ui = min(i+wx, n_rows)
    lj = max(j-wy, 0)
    uj = min(j+wy, n_cols)
    return f(x[li:ui, lj:uj, :])
  return [[window_fn(i,j) 
           for j in np.arange(n_cols)] 
           for i in np.arange(n_rows)] 


