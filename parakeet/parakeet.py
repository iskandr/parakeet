import config
import lib_simple

from adverb_api import allpairs, each, reduce, scan
from lib_simple import *
from prims import *
from run_function import run, specialize_and_compile


def typed_repr(fn, args):
  _, typed, _, _ = specialize_and_compile(fn, args)
  return typed

# TODO: Not sure using builtin names is a good idea.
#def par_sum(x):
"""
  def sum(x):
  return reduce(add, x[1:], init=x[0])
"""
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

class jit:
  def __init__(self, f):
    self.f = f

  def __call__(self, *args, **kwargs):
    return run(self.f, args, kwargs)

def clear_specializations():
  import closure_type
  for clos_t in closure_type._closure_type_cache.itervalues():
    clos_t.specializations.clear()
