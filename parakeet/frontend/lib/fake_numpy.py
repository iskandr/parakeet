import __builtin__

from loopjit.ndtypes import TypeValueT, ScalarT  

from .. decorators import jit, macro
from ... syntax import one_i64, zero_i64
from ... syntax import Range, Return, Cast, UntypedFn, TypedFn, AllocArray
from ... syntax import Tuple, DelayUntilTyped   

from core import float64

@macro
def arange(n, *xs):
  count = __builtin__.len(xs)
  assert 0 <= count <= 2, "Too many args for range: %s" % ((n,) + tuple(xs))
  if count == 0:
    return Range(zero_i64, n, one_i64)
  elif count == 1:
    return Range(n, xs[0], one_i64)  
  else:
    return Range(n, xs[0], xs[1])
 
 
@macro
def empty(shape, dtype = float64):

  def typed_empty(shape, dtype):
    # HACK! 
    # In addition to TypeValue, allow casting functions 
    # to be treated as dtypes 
    if isinstance(dtype, UntypedFn):
      assert len(dtype.body) == 1
      stmt = dtype.body[0]
      assert stmt.__class__ is Return 
      expr = stmt.value 
      assert expr.__class__ is Cast
      elt_t = expr.type 
    elif isinstance(dtype, TypedFn):
      elt_t = dtype.return_type
    else:
      assert isinstance(dtype.type, TypeValueT), \
         "Invalid dtype %s " % (dtype,)
      elt_t = dtype.type.type 
    assert isinstance(elt_t, ScalarT), \
       "Array element type %s must be scalar" % (elt_t,)  
    if isinstance(shape, ScalarT):
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
