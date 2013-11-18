from .. import prims 

from .. frontend import translate_function_value, jit, macro, typed_macro, axis_macro 
from .. ndtypes import make_tuple_type, TupleT, ArrayT, Int64 
from .. syntax import (Map, Tuple,  Array, Attribute, 
                       TupleProj,  const_int, Zip, Len, Reduce)
from ..syntax.helpers import none, false, true 

import numpy as np 
from adverbs import reduce, map 




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
  
