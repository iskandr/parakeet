from .. import prims 
from .. frontend import translate_function_value, jit, macro 
from .. ndtypes import make_tuple_type, TupleT, ArrayT, Int64 
from .. syntax import (Map, Tuple, DelayUntilTyped, Array, Attribute, 
                       TupleProj,  const_int)
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

@jit 
def _tuple_from_args(*args):
  return args

@macro
def builtin_zip(*args):

  elt_tupler = translate_function_value(_tuple_from_args)
  return Map(fn = elt_tupler, args = args)

@macro 
def builtin_tuple(x):
  def typed_tuple(xt):
    if isinstance(xt.type, TupleT):
      return xt 
    else:
      assert isinstance(xt.type, ArrayT), "Can't create type from %s" % (xt.type,)
      assert isinstance(xt, Array), "Can only create tuple from array of const length"
      elt_types = [e.type for e in xt.elts]
      tuple_t = make_tuple_type(elt_types)
      return Tuple(xt.elts, type = tuple_t)
  return DelayUntilTyped(x, typed_tuple)


@macro 
def builtin_len(x):
  def typed_alen(xt):
    if isinstance(xt.type, ArrayT):
      shape = Attribute(xt, 'shape', type = xt.type.shape_t)
      return TupleProj(shape, 0, type = Int64)
    else:
      assert isinstance(xt.type, TupleT), "Can't get 'len' of object of type %s" % xt.type 
      return const_int(len(xt.type.elt_types))
  return DelayUntilTyped(x, typed_alen)

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
  


