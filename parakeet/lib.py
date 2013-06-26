"""Simple library functions which don't depend on adverbs"""
import __builtin__

import syntax
import syntax_helpers

import numpy as np 
import prims 
from decorators import macro, staged_macro, jit

import array_type
from array_type import ArrayT
import core_types 
from core_types import Int8, Int16, Int32, Int64
from core_types import Float32, Float64
from core_types import UInt8, UInt16, UInt32, UInt64
from core_types import Bool
from syntax import Map, AllPairs, Reduce, Scan, IndexMap, IndexReduce
from syntax import DelayUntilTyped, Cast, Range, Attribute, AllocArray
from syntax import Tuple, TupleProj, ArrayView, Index
from syntax_helpers import zero_i64, one_i64, one_i32, const_int 
from tuple_type import empty_tuple_t, TupleT 

@jit 
def identity(x):
  return x

@macro 
def parfor(shape, fn):
  return syntax.ParFor(fn = fn, shape = shape)

@staged_macro("axis")
def map(f, *args, **kwds):
  axis = kwds.get('axis', syntax_helpers.zero_i64)
  return Map(fn = f, args = args, axis = axis)

each = map

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
def ireduce(combine, shape, map_fn = identity, init = None):
  return IndexReduce(fn = map_fn, combine=combine, shape = shape, init = init)

@jit 
def _tuple_from_args(*args):
  return args

@macro
def zip(*args):
  import ast_conversion 
  elt_tupler = ast_conversion.translate_function_value(_tuple_from_args)
  return Map(fn = elt_tupler, args = args)

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
def bool(x):
  return Cast(x, type = Bool)

@jit
def real(x):
  """
  For now we don't have complex types, so real is just the identity function
  """
  return x 

@macro 
def _builtin_tuple(x):
  def typed_tuple(xt):
    if isinstance(xt.type, TupleT):
      return xt 
    else:
      assert isinstance(xt.type, ArrayT), "Can't create type from %s" % (xt.type,)
      assert isinstance(xt, syntax.Array), "Can only create tuple from array of const length"
      elt_types = [e.type for e in xt.elts]
      import tuple_type
      tuple_t = tuple_type.make_tuple_type(elt_types)
      return syntax.Tuple(xt.elts, type = tuple_t)
  return DelayUntilTyped(x, typed_tuple)

@macro 
def alen(x):
  def typed_alen(xt):
    if isinstance(xt.type, ArrayT):
      shape = Attribute(xt, 'shape', type = xt.type.shape_t)
      return TupleProj(shape, 0, type = Int64)
    else:
      assert isinstance(xt.type, TupleT), "Can't get 'len' of object of type %s" % xt.type 
      return const_int(len(xt.type.elt_types))
  return DelayUntilTyped(x, typed_alen)

@macro 
def shape(x):
  def typed_shape(xt):
    if isinstance(xt.type, ArrayT):
      return Attribute(xt, 'shape', type = xt.type.shape_t)
    else:
      return Tuple((), type = empty_tuple_t)
    
  return DelayUntilTyped(x, typed_shape)

@jit 
def sum(x, axis = None):
  return reduce(prims.add, x, init = 0, axis = axis)

@jit 
def prod(x, axis=None):
  return reduce(prims.multiply, x, init=1, axis = axis)

@jit 
def mean(x, axis = None):
  return sum(x, axis = axis) / x.shape[0]

@jit 
def cumsum(x, axis = None):
  return scan(prims.add, x, axis = axis)

@jit 
def cumprod(x, axis = None):
  return scan(prims.multiply, x, axis = axis)

@jit 
def or_(x, y):
  return x or y

@jit 
def and_(x, y):
  return x and y

@jit 
def any(x, axis=None):
  return reduce(or_, x, axis = axis, init = False)

@jit
def all(x, axis = None):
  return reduce(and_, x, axis = axis, init = True)


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
  return reduce(prims.minimum, x, axis = axis)

@jit 
def _builtin_min(x, y = None):
  if y is None:
    return min(x)
  else:
    return prims.minimum(x,y)

@jit 
def max(x, axis = None):
  return reduce(prims.maximum, x, axis = axis)

@jit
def _builtin_max(x, y = None):
  if y is None:
    return max(x)
  else:
    return prims.maximum(x,y)
  
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
def empty(shape, dtype = float64):

  def typed_empty(shape, dtype):
    # HACK! 
    # In addition to TypeValue, allow casting functions 
    # to be treated as dtypes 
    if isinstance(dtype, syntax.Fn):
      assert len(dtype.body) == 1
      stmt = dtype.body[0]
      assert stmt.__class__ is syntax.Return 
      expr = stmt.value 
      assert expr.__class__ is syntax.Cast
      elt_t = expr.type 
    elif isinstance(dtype, syntax.TypedFn):
      elt_t = dtype.return_type
    else:
      assert isinstance(dtype.type, core_types.TypeValueT), \
         "Invalid dtype %s " % (dtype,)
      elt_t = dtype.type.type 
    assert isinstance(elt_t, core_types.ScalarT), \
       "Array element type %s must be scalar" % (elt_t,)  
    if isinstance(shape, core_types.ScalarT):
      shape = Tuple((shape,))
    rank = len(shape.type.elt_types)
    arr_t = array_type.make_array_type(elt_t, rank)
    return AllocArray(shape = shape, elt_type = elt_t, type = arr_t)
  return DelayUntilTyped(values=(shape,dtype), fn = typed_empty) 

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
  def typed_transpose(xt):
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
  return DelayUntilTyped(x, typed_transpose)   

@macro 
def ravel(x):
  return syntax.Ravel(x)
  
@macro 
def reshape(x):
  return syntax.Reshape(x)

@macro 
def elt_type(x):
  def typed_elt_type(xt):
    return syntax.TypeValue(array_type.elt_type(xt.type))
  return DelayUntilTyped(x, typed_elt_type)

@macro
def itemsize(x):
  def typed_itemsize(xt):
    return const_int(array_type.elt_type(xt.type).nbytes)
  return DelayUntilTyped(x, typed_itemsize)

@macro 
def rank(x):
  def typed_rank(xt):
    return const_int(xt.type.rank) 
  return DelayUntilTyped(x, typed_rank)
@macro 
def size(x):
  def typed_size(xt):
    if isinstance(xt.type, array_type.ArrayT):
      return Attribute(xt, 'size', type = Int64)
    else:
      return const_int(1)
  return DelayUntilTyped(x, typed_size)

@jit 
def fill(x, v):
  for i in range(len(x)):
    x[i] = v 
    
    
@jit
def argmax(x):
  """
  Currently assumes axis=None
  TODO: 
    - Support axis arguments
    - use IndexReduce instead of explicit loop
  
      def argmax_map(curr_idx):
        return curr_idx, x[curr_idx]
  
      def argmax_combine((i1,v1), (i2,v2)):
        if v1 > v2:
          return (i1,v1)
        else:
          return (i2,v2)
    
      return ireduce(combine=argmin_combine, shape=x.shape, map_fn=argmin_map, init = (0,x[0]))
  """
  bestval = x[0]
  bestidx = 0
  for i in xrange(1, len(x)):
    currval = x[i]
    if currval > bestval:
      bestval = currval
      bestidx = i
  return bestidx 

@jit
def argmin(x):
  """
  Currently assumes axis=None
  TODO: 
    - Support axis arguments
    - use IndexReduce instead of explicit loop
  
      def argmin_map(curr_idx):
        return curr_idx, x[curr_idx]
  
      def argmin_combine((i1,v1), (i2,v2)):
        if v1 < v2:
          return (i1,v1)
        else:
          return (i2,v2)
    
      return ireduce(combine=argmin_combine, shape=x.shape, map_fn=argmin_map, init = (0,x[0]))
  """
  bestval = x[0]
  bestidx = 0
  for i in xrange(1, len(x)):
    currval = x[i]
    if currval < bestval:
      bestval = currval
      bestidx = i
  return bestidx 

@jit
def copy(x):
  return [xi for xi in x]


@jit
def pmap1(f, x, w = 3):
  n = x.shape[0]
  def local_apply(i):
    lower = __builtins__.max(i-w/2, 0)
    upper = __builtins__.min(i+w/2+1, n)
    elts = x[lower:upper]
    return f(elts)
  return imap(local_apply, n)

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
    lx = __builtins__.max(i-hx, 0)
    ux = __builtins__.min(i+hx+1, n_rows)
    ly = __builtins__.max(j-hy, 0)
    uy = __builtins__.min(j+hy+1, n_cols)
    return f(x[lx:ux, ly:uy])
    
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



