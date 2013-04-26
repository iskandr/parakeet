"""Simple library functions which don't depend on adverbs"""

from prims import *
from adverb_api import allpairs, each, reduce, scan, conv
from decorators import macro, staged_macro
from core_types import Int64, Float64
import syntax_helpers

def identity(x):
  return x

def len(arr):
  return arr.shape[0]

def shape(arr):
  return arr.shape

def sum(x, axis = None):
  return reduce(add, x, init = 0, axis = axis)

def prod(x, axis=None):
  return reduce(multiply, x[1:], init=x[0], axis = axis)

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
  return reduce(minimum, x[1:], init=x[0], axis = axis)

def min(x, y = None):
  if y is None:
    return minelt(x)
  else:
    return minimum(x,y)

def maxelt(x, axis = None):
  return reduce(maximum, x[1:], init=x[0], axis = axis)

def max(x, y = None):
  if y is None:
    return maxelt(x)
  else:
    return maximum(x,y)
  
@macro 
def range1(n):
  return syntax.Range(syntax_helpers.zero_i64, n, syntax_helpers.one_i64)

@macro 
def range2(start, stop):
  return syntax.Range(start, stop, syntax_helpers.one_i64)

@macro 
def range3(start, stop, step):
  return syntax.Range(start, stop, step)

def range(a, b = None, c = None):
  if c is None:
    if b is None:
      return range1(a)
    else:
      return range2(a,b)
  else:
    return range3(a,b,c)

@macro
def int(x):
  return syntax.Cast(x, type=Int64)

@macro
def long(x):
  return syntax.Cast(x, type=Int64)

@macro
def float(x):
  return syntax.Cast(x, type=Float64)

@macro 
def zeros_like(x):
  return syntax.ConstArrayLike(x, 0)

@macro
def ones_like(x):
  return syntax.ConstArrayLike(x, 1)
