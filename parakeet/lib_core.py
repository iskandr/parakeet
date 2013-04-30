"""Simple library functions which don't depend on adverbs"""
import __builtin__


import syntax
import syntax_helpers

from prims import *
from adverb_api import allpairs, each, reduce, scan
from decorators import macro, staged_macro
from core_types import Int8, Int16, Int32, Int64
from core_types import Float32, Float64
from core_types import UInt8, UInt16, UInt32, UInt64
from core_types import Bool
from syntax_helpers import zero_i64, one_i64, one_i32

@macro
def map(f, *args, **kwds):
  axis = kwds.get('axis', syntax_helpers.none)
  return syntax.Map(fn = f, )
  

@macro 
def int8(x):
  return syntax.Cast(x, type = Int8) 

@macro 
def int16(x):
  return syntax.Cast(x, type = Int16) 

@macro 
def int32(x):
  return syntax.Cast(x, type = Int32) 

@macro 
def int64(x):
  return syntax.Cast(x, type = Int64) 

int = int64 
long = int64 

@macro 
def uint8(x):
  return syntax.Cast(x, type = UInt8) 

@macro 
def uint16(x):
  return syntax.Cast(x, type = UInt16) 

@macro 
def uint32(x):
  return syntax.Cast(x, type = UInt32) 

@macro 
def uint64(x):
  return syntax.Cast(x, type = UInt64)

uint = uint64 

@macro 
def float32(x):
  return syntax.Cast(x, type = Float32)

@macro 
def float64(x):
  return syntax.Cast(x, type = Float64)

float = float64

def bool8(x):
  return syntax.Cast(x, type = Bool)

bool = bool8 

def identity(x):
  return x

def len(arr):
  return arr.shape[0]

def shape(arr):
  return arr.shape

def sum(x, axis = None):
  return reduce(add, x, init = 0, axis = axis)

def prod(x, axis=None):
  return reduce(multiply, x, init=1, axis = axis)

def mean(x, axis = None):
  return sum(x, axis = axis) / x.shape[0]

def cumsum(x, axis = None):
  return scan(add, x, axis = axis)

def cumprod(x, axis = None):
  return scan(multiply, x, axis = axis)

def diff(x):
  """
  TODO:
    - axis selection
    - preserve size by filling with zeros
    - allow n'th differences by recursion
  """
  return x[1:] - x[:-1]

def dot(x,y):
  return sum(x*y)


def minelt(x, axis = None):
  return reduce(minimum, x, axis = axis)

def min(x, y = None):
  if y is None:
    return minelt(x)
  else:
    return minimum(x,y)

def maxelt(x, axis = None):
  return reduce(maximum, x, axis = axis)

def max(x, y = None):
  if y is None:
    return maxelt(x)
  else:
    return maximum(x,y)
  

@macro
def range(n, *xs):
  count = __builtin__.len(xs)
  assert 0 <= count <= 2, "Too many args for range: %s" % ((n,) + tuple(xs))
  if count == 0:
    return syntax.Range(syntax_helpers.zero_i64, n, syntax_helpers.one_i64)
  elif count == 1:
    return syntax.Range(n, xs[0], syntax_helpers.one_i64)  
  else:
    return syntax.Range(n, xs[0], xs[1])
arange = range 

@macro
def fill(shape, fn):
  return syntax.Fill(shape = shape, fn = fn)

def zeros(shape, dtype = float64):
  zero = dtype(0)
  return fill(shape, lambda _: zero)

def zeros_like(x, dtype = float64):
  return zeros(x.shape, dtype)

def ones(shape, dtype = float64):
  one = dtype(1)
  return fill(shape, lambda _: one)

def ones_like(x):
  return ones(x.shape)



