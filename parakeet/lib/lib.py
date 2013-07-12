"""Simple library functions which don't depend on adverbs"""
import __builtin__

import syntax
import syntax_helpers

import numpy as np 
import prims 
from frontend import macro, staged_macro, jit

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
def real(x):
  """
  For now we don't have complex types, so real is just the identity function
  """
  return x 



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
  
