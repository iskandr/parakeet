"""Simple library functions which don't depend on adverbs"""

from prims import *
from adverb_api import allpairs, each, reduce, scan, conv
from decorators import macro, staged_macro

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
  return reduce(min, x[1:], init=x[0], axis = axis)

def maxelt(x, axis = None):
  return reduce(max, x[1:], init=x[0], axis = axis)

@macro 
def zeros_like(x):
  return syntax.ConstArrayLike(x, 0)

@macro
def ones_like(x):
  return syntax.ConstArrayLike(x, 1)
