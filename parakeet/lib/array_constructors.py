import __builtin__
import numpy as np 

from .. frontend.decorators import jit, macro, typed_macro 
from .. ndtypes import ( make_array_type,  combine_type_list, repeat_tuple,  
                         ArrayT, TupleT,  Int64, ScalarT) 
from .. syntax import (Range, Cast, AllocArray, TupleProj, Array, 
                       ConstArray, ConstArrayLike, Index, Shape)

from ..syntax.helpers import get_types, one_i64, zero_i64, make_tuple

from adverbs import imap 
from numpy_types import float64
from lib_helpers import _get_type, _get_shape, _get_tuple_elts


@macro
def arange(n, *xs, **kwds):
  if 'dtype' in kwds:
    elt_t = _get_type(kwds['dtype'])
    del kwds['dtype']
  else:
    elt_t = Int64
    
  assert len(kwds) == 0, "Unexpected keyword arguments to 'arange': %s" % kwds
  array_t = make_array_type(elt_t, 1) 
  count = __builtin__.len(xs)
  assert 0 <= count <= 2, "Too many args for range: %s" % ((n,) + tuple(xs))
 
  if count == 0:
    start = zero_i64 
    stop = n 
    step = one_i64 
  elif count == 1:
    start = n 
    stop = xs[0]
    step = one_i64 
  else:
    start = n 
    stop = xs[0]
    step = xs[1]
    
  if elt_t != Int64:
    start = Cast(start, type = elt_t)
    stop = Cast(stop, type = elt_t)
    step = Cast(step, type = elt_t)
  return Range(start, stop, step, type = array_t)
 
@typed_macro
def empty(shape, dtype = float64):
  elt_t = _get_type(dtype)
  assert isinstance(elt_t, ScalarT), "Array element type %s must be scalar" % (elt_t,)
  shape = _get_shape(shape)  
  rank = len(shape.type.elt_types)
  arr_t = make_array_type(elt_t, rank)
  return AllocArray(shape = shape, elt_type = elt_t, type = arr_t)

@jit 
def empty_like(x, dtype = None):
  if dtype is None:
    return empty(x.shape, x.dtype)
  else:
    return empty(x.shape, dtype)
  
@typed_macro   
def zeros(shape, dtype = float64):
  shape = _get_shape(shape)
  elt_type = _get_type(dtype)
  zero = Cast(zero_i64, type = elt_type)
  ndims = len(shape.type.elt_types)
  if ndims == 0:
    return zero 
  else:
    t = make_array_type(elt_type, ndims)
    return ConstArray(shape = shape, value = zero, type = t)
  
@jit
def zeros_like(x, dtype = None):
  if dtype is None:
    dtype = x.dtype
  return zeros(x.shape, dtype)

@typed_macro
def ones(shape, dtype = float64):
  shape = _get_shape(shape)
  elt_type = _get_type(dtype)
  one = Cast(one_i64, type = elt_type)
  ndims = len(shape.type.elt_types)
  if ndims == 0:
    return one 
  else:
    t = make_array_type(elt_type, ndims)
    return ConstArray(shape = shape, value = one, type = t)

@jit  
def ones_like(x, dtype = None):

  if dtype is None:
    dtype = x.dtype
  return ones(x.shape, dtype)

@jit
def copy(x):
  return [xi for xi in x]

@jit 
def linspace(start, stop, num = 50, endpoint = True):
  """
  Copied from numpy.linspace but dropped the 'retstep' option 
  which allows you to optionall return a tuple (that messes with type inference)
  """
  num = int(num)
  if num <= 0:
    return np.array([])
  elif endpoint:
    if num == 1:
      return np.array([float(start)])
    step = (stop-start)/float((num-1))
    y = np.arange(0, num) * step + start
    y[-1] = stop
    return y
  else:
    step = (stop-start)/float(num)
    return np.arange(0, num) * step + start 



@typed_macro
def array(value, dtype = None):
  if dtype is not None:
    expected_elt_type = _get_type(dtype)
    print "Got dtype argument to array: %s, ignoring for now" % expected_elt_type

  if isinstance(value.type, ArrayT):
    return value 
  else:
    assert isinstance(value.type, TupleT), "Don't know how to make array from %s : %s" % (value, value.type)
    elt_types = value.type.elt_types
    assert all(isinstance(t, ScalarT) for t in elt_types), \
      "Can only make array from tuple of scalars, not %s : %s" % (value, value.type)
    elt_t = combine_type_list(value.type.elt_types)
    array_elts = _get_tuple_elts(value, cast_type = elt_t)
    array_t = make_array_type(elt_t, 1)
    return Array(elts = array_elts, type = array_t)

@typed_macro
def tile(A, reps):
  reps = _get_shape(reps)
  reps_dims = len(reps.type.elt_types)
  if reps_dims == 0:
    return A 
  A_rank = A.type.rank if isinstance(A.type, ArrayT) else 0 
  if A_rank == 0:
    # array scalars, ugh!
    if isinstance(A.type, ArrayT):
      A = Index(A, make_tuple(()), type = A.elt_type)
    assert isinstance(A.type, ScalarT), "First argument to 'tile' must be array or scalar"
    array_t = make_array_type(A.type, reps_dims)
    return ConstArray(value = A, shape = reps, type = array_t)
  else:
    A_shape = Shape(A, type = repeat_tuple(Int64, A_rank))
    A_shape_elts = _get_tuple_elts(A_shape)
    reps_elts = _get_tuple_elts(reps)
    result_shape_elts = []
    assert False, "np.tile not yet implemented"
    #for i in xrange(max(len(A_shape_elts), len(reps_elts))):
      
    #result = AllocArray()
    
  