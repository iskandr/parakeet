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
def winmap1d(f, x, w = 3):
  n = x.shape[0]
  h = w / 2
  def window_fn(i):
    lower = max(i-h, 0)
    upper = min(i+h, n)
    return f(x[lower:upper])
  return [window_fn(i) for i in np.arange(n)]
        
@jit  
def winmap2d(f, x, wx = 3, wy = 3):
  n_rows, n_cols = x.shape
  hx = wx / 2
  hy = wy / 2
  def window_fn(i,j):
    li = max(i-hx, 0)
    ui = min(i+hx, n_rows)
    lj = max(j-hy, 0)
    uj = min(j+hy, n_cols)
    return f(x[li:ui, lj:uj])
  return [[window_fn(i,j) 
           for j in np.arange(n_cols)] 
           for i in np.arange(n_rows)] 
  


