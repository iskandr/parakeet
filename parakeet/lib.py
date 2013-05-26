"""Simple library functions which don't depend on adverbs"""
import __builtin__

import syntax
import syntax_helpers

from prims import *
from decorators import macro, staged_macro, jit

import array_type
from array_type import ArrayT
from core_types import Int8, Int16, Int32, Int64
from core_types import Float32, Float64
from core_types import UInt8, UInt16, UInt32, UInt64
from core_types import Bool, TypeValueT
from syntax import Map, AllPairs, Reduce, Scan, IndexMap, IndexReduce
from syntax import DelayUntilTyped, Cast, Range, Attribute, AllocArray
from syntax import Tuple, TupleProj, ArrayView
from syntax_helpers import zero_i64, one_i64, one_i32, const_int 
from tuple_type import empty_tuple_t 

@jit 
def identity(x):
  return x

@staged_macro("axis")
def map(f, *args, **kwds):
  axis = kwds.get('axis', syntax_helpers.zero_i64)
  return Map(fn = f, args = args, axis = axis)

@staged_macro("axis") 
def allpairs(f, x, y, axis = 0):
  return AllPairs(fn = f, args = (x,y), axis = axis)

@staged_macro("axis")
def reduce(f, *args, **kwds):
  axis = kwds.get('axis', syntax_helpers.none)
  init = kwds.get('init', syntax_helpers.none)
  import ast_conversion
  ident = ast_conversion.translate_function_value(identity)
  return Reduce(fn = ident, 
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
  return Scan(fn = ident,  
                     combine = f,
                     emit = ident, 
                     args = args,
                     init = init,
                     axis = axis)

@macro
def imap(fn, shape):
  return IndexMap(shape = shape, fn = fn)

@macro
def ireduce(fn, shape, init = None):
  return IndexReduce(fn = fn, shape = shape, init = init)

@macro 
def int8(x):
  return Cast(x, type = Int8) 

@macro 
def int16(x):
  return Cast(x, type = Int16) 

@macro 
def int32(x):
  return Cast(x, type = Int32) 

@macro 
def int64(x):
  return Cast(x, type = Int64) 


@macro 
def uint8(x):
  return Cast(x, type = UInt8) 

@macro 
def uint16(x):
  return Cast(x, type = UInt16) 

@macro 
def uint32(x):
  return Cast(x, type = UInt32) 

@macro 
def uint64(x):
  return Cast(x, type = UInt64)

uint = uint64 

@macro 
def float32(x):
  return Cast(x, type = Float32)

@macro 
def float64(x):
  return Cast(x, type = Float64)

@macro 
def bool8(x):
  return Cast(x, type = Bool)

@jit
def real(x):
  """
  For now we don't have complex types, so real is just the identity function
  """
  return x 

@jit 
def alen(arr):
  return arr.shape[0]

@macro 
def shape(x):
  def fn(xt):
    if isinstance(xt.type, ArrayT):
      return Attribute(xt, 'shape', type = xt.type.shape_t)
    else:
      return Tuple((), type = empty_tuple_t)
    
  return DelayUntilTyped(values = (x,), fn = fn)

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
def or_(x, y):
  return x or y

@jit 
def and_(x, y):
  return x and y

@jit 
def any(x, axis=None):
  return reduce(x, or_, axis = axis)

@jit
def all(x, axis = None):
  return reduce(x, and_, axis = axis)


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
def min(x, axis = None):
  return reduce(minimum, x, axis = axis)

@jit 
def _prim_min(x, y = None):
  if y is None:
    return min(x)
  else:
    return minimum(x,y)

@jit 
def max(x, axis = None):
  return reduce(maximum, x, axis = axis)

@jit
def _prim_max(x, y = None):
  if y is None:
    return max(x)
  else:
    return maximum(x,y)
  
@macro
def arange(n, *xs):
  count = __builtin__.len(xs)
  assert 0 <= count <= 2, "Too many args for range: %s" % ((n,) + tuple(xs))
  if count == 0:
    return Range(syntax_helpers.zero_i64, n, syntax_helpers.one_i64)
  elif count == 1:
    return Range(n, xs[0], syntax_helpers.one_i64)  
  else:
    return Range(n, xs[0], xs[1])
 
 
@macro
def empty(shape, dtype):
  return AllocArray(shape = shape, elt_type = dtype) 

@jit 
def empty_like(x, dtype = None):
  if dtype is None:
    return empty(x.shape, x.dtype)
  else:
    return empty(x.shape, dtype)
  
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

@macro
def transpose(x):
  def fn(xt):
    if isinstance(xt.type, ArrayT) and xt.type.rank > 1:
      shape = Attribute(xt, 'shape', type = xt.type.shape_t)
      strides = Attribute(xt, 'strides', type = xt.type.strides_t)
      data = Attribute(xt, 'data', type = xt.type.ptr_t)
      size = Attribute(xt, 'size', type = Int64)
      offset = Attribute(xt, 'offset', type = Int64)
      ndims = xt.type.rank 
      shape_elts = [TupleProj(shape, i, type = Int64) 
                               for i in xrange(ndims)]
      stride_elts = [TupleProj(strides, i, type = Int64) 
                                 for i in xrange(ndims)]
      new_shape = Tuple(tuple(reversed(shape_elts)))
      new_strides = Tuple(tuple(reversed(stride_elts)))
      return ArrayView(data, 
                       new_shape, 
                       new_strides, 
                       offset, 
                       size, 
                       type = xt.type)
    else:
      return xt 
  return DelayUntilTyped(values = (x,), fn = fn)   
  #

@macro 
def ravel(x):
  return syntax.Ravel(x)

@macro 
def reshape(x):
  return syntax.Reshape(x)

@macro 
def elt_type(x):
  return DelayUntilTyped(
    values = (x,), 
    fn = lambda xt: TypeValueT(array_type.elt_type(xt.type)))

@macro
def itemsize(x):
  return DelayUntilTyped(
    values = (x,), 
    fn = lambda xt: const_int(array_type.elt_type(xt.type).nbytes))

@macro 
def rank(x):
  return DelayUntilTyped(
    values = (x,), 
    fn = lambda xt: const_int(xt.type.rank))

@macro 
def size(x):
  def fn(xt):
    if isinstance(xt.type, array_type.ArrayT):
      return Attribute(xt, 'size', type = Int64)
    else:
      return const_int(1)
  return DelayUntilTyped(values = (x,), fn = fn)

@jit 
def fill(x, v):
  for i in range(len(x)):
    x[i] = v 
    
@jit
def argmax(x):
  """
  Currently assumes axis=None
  TODO: Support axis arguments
  """
  def helper(curr_idx, acc):
    max_val = acc[1]
    v = x[curr_idx]
    if v > max_val:
      return (curr_idx, v)
    else:
      return acc
  return ireduce(helper, x.shape, init = (0,x[0]))

@jit
def argmin(x):
  """
  Currently assumes axis=None
  TODO: Support axis arguments
  """
  def helper(curr_idx, acc):
    min_val = acc[1]
    v = x[curr_idx]
    if v < min_val:
      return (curr_idx, v)
    else:
      return acc
  return ireduce(helper, x.shape, init = (0,x[0]))

@jit
def copy(x):
  return [xi for xi in x]


@jit
def pmap1(f, x, w = 3):
  n = x.shape[0]
  h = w / 2
  return [f(x[max(i-h, 0):min(i+h+1, n)]) for i in range(n)]


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
    if i <= hx or i >= n_rows - hx or j < hy or j >= n_cols - hy:
      lx = max(i-hx, 0)
      ux = min(i+hx+1, n_rows)
      ly = max(j-hy, 0)
      uy = min(j+hy+1, n_cols)
      result = f(x[lx:ux, ly:uy])
    else:
      lx = i-hx
      ux = i+hx+1
      ly = j-hy
      uy = j+hy+1
      result = f(x[lx:ux, ly:uy])
    return result
    
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



