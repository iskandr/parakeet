
from .. import prims
from .. ndtypes import ArrayT, Int64, elt_type, empty_tuple_t
from .. frontend import macro, jit 
from .. syntax import (Attribute, TupleProj, ArrayView, DelayUntilTyped, 
                       Tuple, Ravel, Reshape, TypeValue, 
                       const_int)

from adverbs import map, reduce, scan 

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
  return Ravel(x)
  
@macro 
def reshape(x):
  return Reshape(x)

@macro 
def elt_type(x):
  def typed_elt_type(xt):
    return TypeValue(elt_type(xt.type))
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


@macro 
def shape(x):
  def typed_shape(xt):
    if isinstance(xt.type, ArrayT):
      return Attribute(xt, 'shape', type = xt.type.shape_t)
    else:
      return Tuple((), type = empty_tuple_t)
    
  return DelayUntilTyped(x, typed_shape)


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
def square(x):
  return x * x 

@jit
def conjugate(x):
  """
  For now we don't have complex numbers so this is just the identity function
  """
  return x 

def _scalar_sign(x):
  return x if x >= 0 else -x 

@jit
def sign(x):
  return map(_scalar_sign(x))

@jit 
def reciprocal(x):
  return 1 / x


@jit 
def rad2deg(rad):
  return rad * 180 / 3.141592653589793

@jit
def deg2rad(deg):
  return deg * 3.141592653589793 / 180 
