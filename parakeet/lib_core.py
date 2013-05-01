"""Simple library functions which don't depend on adverbs"""
import __builtin__

import syntax
import syntax_helpers

from prims import *
from decorators import macro, staged_macro, jit 
from core_types import Int8, Int16, Int32, Int64
from core_types import Float32, Float64
from core_types import UInt8, UInt16, UInt32, UInt64
from core_types import Bool
from syntax_helpers import zero_i64, one_i64, one_i32

@jit 
def identity(x):
  return x

@staged_macro("axis")
def map(f, *args, **kwds):
  axis = kwds.get('axis', syntax_helpers.zero_i64)
  return syntax.Map(fn = f, args = args, axis = axis)

each = map

@staged_macro("axis") 
def allpairs(f, x, y, axis = 0):
  return syntax.AllPairs(fn = f, args = (x,y), axis = axis)

@staged_macro("axis")
def reduce(f, *args, **kwds):
  axis = kwds.get('axis', syntax_helpers.none)
  init = kwds.get('init', syntax_helpers.none)
  import ast_conversion
  ident = ast_conversion.translate_function_value(identity)
  return syntax.Reduce(fn = ident, 
                       combine = f, 
                       args = args,
                       init = init,
                       axis = axis)

@staged_macro("axis")
def scan(f, *args, **kwds):
  axis = kwds.get('axis', syntax_helpers.zero_i64)
  init = kwds.get('init', syntax_helpers.none)
  import ast_conversion
  ident = ast_conversion.translate_function_value(identity)
  return syntax.Scan(fn = ident,  
                     combine = f,
                     emit = ident, 
                     args = args,
                     init = init,
                     axis = axis)

@macro
def imap(fn, shape):
  return syntax.IndexMap(shape = shape, fn = fn)

@macro
def ireduce(fn, shape, init = None):
  return syntax.IndexReduce(fn = fn, shape = shape, init = init)

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

@macro 
def bool8(x):
  return syntax.Cast(x, type = Bool)

bool = bool8 
bool_ = bool8 

@jit 
def len(arr):
  return arr.shape[0]

@jit 
def shape(arr):
  return arr.shape

@jit 
def sum(x, axis = None):
  return reduce(add, x, init = 0, axis = axis)

@jit 
def prod(x, axis=None):
  return reduce(multiply, x, init=1, axis = axis)

@jit 
def mean(x, axis = None):
  return sum(x, axis = axis) / x.shape[0]

@jit 
def cumsum(x, axis = None):
  return scan(add, x, axis = axis)

@jit 
def cumprod(x, axis = None):
  return scan(multiply, x, axis = axis)

@jit 
def diff(x):
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

@jit 
def minelt(x, axis = None):
  return reduce(minimum, x, axis = axis)

@jit 
def min(x, y = None):
  if y is None:
    return minelt(x)
  else:
    return minimum(x,y)

@jit 
def maxelt(x, axis = None):
  return reduce(maximum, x, axis = axis)

@jit
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

#@macro
#def empty(shape, dtype = np.float64):
#  return syntax.Alloc(elt_type = dtype, shape = shape)

@jit 
def zeros(shape, dtype = float64):
  zero = dtype(0)
  return imap(lambda _: zero, shape)

@jit
def zeros_like(x, dtype = None):
  if dtype is None:
    dtype = x.dtype
  return zeros(x.shape, dtype)

@jit
def ones(shape, dtype = float64):
  one = dtype(1)
  return imap(lambda _: one, shape)

@jit
def ones_like(x, dtype = None):
  if dtype is None:
    dtype = x.dtype
  return ones(x.shape)