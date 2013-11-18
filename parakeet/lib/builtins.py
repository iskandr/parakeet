from .. import prims 

from .. frontend import jit,  typed_macro 
from .. ndtypes import make_tuple_type, TupleT, ArrayT
from ..syntax import Tuple, Array 

@jit 
def builtin_or(x, y):
  return x or y

@jit 
def builtin_and(x, y):
  return x and y


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
  
