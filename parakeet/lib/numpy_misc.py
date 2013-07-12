
from .. frontend import macro, jit 

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
