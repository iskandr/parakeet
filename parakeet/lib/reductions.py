
from ..frontend import jit, axis_macro 
from .. import prims 
from adverbs import scan 

from .. frontend import translate_function_value

from .. syntax import Reduce, Const 
from ..syntax.helpers import none, false, true, one_i32, zero_i32, zero_i24
 
from adverbs import reduce
from builtins import builtin_and, builtin_or
from parakeet.syntax.delay_until_typed import DelayUntilTyped

def _identity(x):
  return x 

def mk_reduce(combiner, x, init, axis):
  return Reduce(fn = translate_function_value(_identity), 
                combine = translate_function_value(combiner), 
                args = (x,),  
                init = init, 
                axis = axis)

@axis_macro
def reduce_min(x, axis = None):
  return mk_reduce(prims.minimum, x, init = None, axis = axis)


@axis_macro
def reduce_max(x, axis = None):
  return mk_reduce(prims.maximum, x, init = None, axis = axis)


@axis_macro 
def reduce_any(x, axis=None):
  return mk_reduce(builtin_or, x, init  = false, axis = axis, )

@axis_macro
def reduce_all(x, axis = None):
  return mk_reduce(builtin_and, x, init = true, axis = axis)

@axis_macro 
def reduce_sum(x, axis = None):
  return mk_reduce(prims.add, x, init = zero_i24, axis = axis)
  
@jit 
def builtin_min(x, y = None):
  if y is None:
    return reduce_min(x)
  else:
    return prims.minimum(x,y)

@jit
def builtin_max(x, y = None):
  if y is None:
    return reduce_max(x)
  else:
    return prims.maximum(x,y)


@jit 
def prod(x, axis=None):
  return reduce(prims.multiply, x, init=true, axis = axis)

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
def vdot(x,y):
  """
  Inner product between two 1Dvectors
  """
  return sum(x*y)

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


