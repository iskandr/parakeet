from .. import prims 
from .. frontend import translate_function_value, jit, macro, typed_macro 
from .. ndtypes import make_tuple_type, TupleT, ArrayT, Int64 
from .. syntax import (Map, Tuple,  Array, Attribute, 
                       TupleProj,  const_int, Zip, Len)
from adverbs import reduce, map 

@jit 
def builtin_or(x, y):
  return x or y

@jit 
def builtin_and(x, y):
  return x and y

@jit 
def builtin_any(x, axis=None):
  return reduce(builtin_or, x, axis = axis, init = False)

@jit
def builtin_all(x, axis = None):
  return reduce(builtin_and, x, axis = axis, init = True)

@jit 
def builtin_sum(x, axis = None):
  return reduce(prims.add, x, init = 0, axis = axis)


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
  

@jit
def reduce_min(x, axis = None):
  return reduce(prims.minimum, x, axis = axis)

@jit 
def builtin_min(x, y = None):
  if y is None:
    return reduce_min(x)
  else:
    return prims.minimum(x,y)

@jit 
def reduce_max(x, axis = None):
  return reduce(prims.maximum, x, axis = axis)

@jit
def builtin_max(x, y = None):
  if y is None:
    return reduce_max(x)
  else:
    return prims.maximum(x,y)
  


