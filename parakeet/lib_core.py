"""Simple library functions which don't depend on adverbs"""
import __builtin__
from prims import *
from adverb_api import allpairs, each, reduce, scan
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



