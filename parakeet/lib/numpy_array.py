import numpy as np 

from .. import prims, ndtypes 
from .. ndtypes import ArrayT, Int64, elt_type, empty_tuple_t
from .. frontend import macro, jit 
from .. syntax import (Attribute, TupleProj, ArrayView, DelayUntilTyped, 
                       Tuple, Ravel, Reshape, TypeValue, 
                       const_int, Transpose)

from adverbs import map, reduce, scan 

@macro
def transpose(x):
  return Transpose(x)

@macro 
def ravel(x):
  return Ravel(x)
  
@macro 
def reshape(x):
  return Reshape(x)

@macro 
def get_elt_type(x):
  def typed_elt_type(xt):
    return TypeValue(ndtypes.elt_type(xt.type))
  return DelayUntilTyped(x, typed_elt_type)

@macro
def itemsize(x):
  def typed_itemsize(xt):
    return const_int(elt_type(xt.type).nbytes)
  return DelayUntilTyped(x, typed_itemsize)

@macro 
def rank(x):
  def typed_rank(xt):
    return const_int(xt.type.rank) 
  return DelayUntilTyped(x, typed_rank)

@macro 
def size(x):
  def typed_size(xt):
    if isinstance(xt.type, ArrayT):
      return Attribute(xt, 'size', type = Int64)
    else:
      return const_int(1)
  return DelayUntilTyped(x, typed_size)

@jit 
def fill(x, v):
  for i in range(len(x)):
    x[i] = v 
 

@macro 
def shape(x):
  def typed_shape(xt):
    if isinstance(xt.type, ArrayT):
      return Attribute(xt, 'shape', type = xt.type.shape_t)
    else:
      return Tuple((), type = empty_tuple_t)
    
  return DelayUntilTyped(x, typed_shape)




@jit 
def diff(x):
  """
  TODO:
    - axis selection
    - preserve size by filling with zeros
    - allow n'th differences by recursion
  """
  return x[1:] - x[:-1]
