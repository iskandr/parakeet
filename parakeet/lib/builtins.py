from .. import prims 

from .. frontend import translate_function_value, jit, macro, typed_macro, axis_macro 
from .. ndtypes import make_tuple_type, TupleT, ArrayT, Int64 
from .. syntax import (Map, Tuple,  Array, Attribute, 
                       TupleProj,  const_int, Zip, Len, Reduce)
from ..syntax.helpers import none, false, true 

import numpy as np 
from adverbs import reduce, map 

def _identity(x):
  return x 

def mk_reduce(combiner, x, init, axis):
  return Reduce(fn = translate_function_value(_identity), 
                combine = combiner, 
                args = (x,),  
                init = init, 
                axis = axis)

@jit 
def builtin_or(x, y):
  return x or y

@jit 
def builtin_and(x, y):
  return x and y

@axis_macro 
def builtin_any(x, axis=None):
  return mk_reduce(builtin_or, x, init  = false, axis = axis, )

@axis_macro
def builtin_all(x, axis = None):
  return mk_reduce(builtin_and, x, init = true, axis = axis)

@axis_macro 
def builtin_sum(x, axis = None):
  #return reduce(prims.add, x, init = False, axis = axis)
  return mk_reduce(prims.add, x, init = false, axis = axis)

@typed_macro 
def builtin_tuple(xt):
  if isinstance(xt.type, TupleT):
    return xt 
  else:
    assert isinstance(xt.type, ArrayT), "Can't create type from %s" % (xt.type,)
    assert isinstance(xt, Array), "Can only create tuple from array of const length"
    elt_types = [e.type for e in xt.elts]
    tuple_t = make_tuple_type(elt_types)
    return Tuple(xt.elts, type = tuple_t)
  

@axis_macro
def reduce_min(x, axis = None):
  return mk_reduce(prims.minimum, x, init = None, axis = axis)

@jit 
def builtin_min(x, y = None):
  if y is None:
    return reduce_min(x)
  else:
    return prims.minimum(x,y)

@axis_macro
def reduce_max(x, axis = None):
  return mk_reduce(prims.maximum, x, init = None, axis = axis)

@jit
def builtin_max(x, y = None):
  if y is None:
    return reduce_max(x)
  else:
    return prims.maximum(x,y)
  


