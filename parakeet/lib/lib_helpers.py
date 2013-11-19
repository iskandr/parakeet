import numpy as np

from ..frontend import macro, jit 
from ..syntax import (UntypedFn, Return, Cast, TypedFn, TypeValue, Array, Tuple, Expr, Closure, 
                      TupleProj) 
from ..syntax.helpers import get_types, make_tuple  
from ..ndtypes import (type_conv, TypeValueT, ArrayT, IntT, 
                       make_tuple_type, TupleT, ScalarT, ClosureT, FnT, Type)


def _get_type(dtype):
  """
  Defensively try to extract a scalar type from any wacky thing
  that might get passed in as a 'dtype' to an array constructor
  """
  if isinstance(dtype, macro):
    dtype = dtype.as_fn()
  
  while isinstance(dtype, jit):
    dtype = dtype.f 
  
  while isinstance(dtype, Expr):
    if isinstance(dtype, UntypedFn):
      if len(dtype.body) == 1:
        stmt = dtype.body[0]
        if stmt.__class__ is Return:
          expr = stmt.value 
          if expr.__class__ is Cast:
            dtype = expr.type 
            break
      assert False, "Don't know how to convert function %s into Parakeet type" % dtype
    elif isinstance(dtype, TypedFn):
      dtype = dtype.return_type
    if isinstance(dtype, TypeValue):
      dtype = dtype.type_value  
    elif isinstance(dtype.type, TypeValueT):
      dtype = dtype.type
    elif isinstance(dtype, Closure):
      dtype = dtype.fn
    elif isinstance(dtype.type, (ClosureT, FnT)):
      dtype = dtype.type.fn
    else:
      assert False, "Don't know how to turn %s : %s into Parakeet type" % (dtype, dtype.type)  


  if isinstance(dtype, Type):
    if isinstance(dtype, TypeValueT):
      return dtype.type
    else:
      return dtype 
    
  elif isinstance(dtype, (np.dtype, type)):

    return type_conv.equiv_type(dtype)
  elif isinstance(dtype, str):
    return type_conv.equiv_type(np.dtype(dtype))
  assert False, "Don't know how to turn %s into Parakeet type" % dtype  

def _get_shape(value):
  """
  User might pass a scalar, a tuple of integer values, or a literal array of integers
  as the 'shape' argument of array constructor. Normalize those all into a tuple
  """
  if isinstance(value.type, ArrayT):
    assert value.__class__ is Array, "Don't know how to convert %s into tuple" % value
    elts = value.elts 
    
    elt_types = get_types(elts)
    assert all(isinstance(t, IntT) for t in elt_types), \
      "Shape elements must be integers, not %s" % elt_types
    return Tuple(elts = elts, type = make_tuple_type(elt_types))
  elif isinstance(value.type, TupleT):
    assert all(isinstance(t, ScalarT) for t in value.type.elt_types), \
      "Shape tuple %s : %s has non-scalar elements" % (value, value.type)
    return value
  elif isinstance(value.type, ScalarT):
    assert isinstance(value.type, IntT), \
      "Can't make shape tuple from non-integer scalar %s : %s" % (value, value.type)
    return make_tuple((value,))
  assert False, "Can't make shape tuple from value %s : %s" % (value, value.type)
  
def _get_tuple_elts(expr, cast_type = None):
  elts = []
  for i, tuple_elt_t in enumerate(expr.type.elt_types):
    if expr.__class__ is Tuple:
      tuple_elt = expr.elts[i]
    else:
      tuple_elt = TupleProj(expr, i, type = tuple_elt_t)
    if cast_type is not None and tuple_elt_t != cast_type:
      tuple_elt = Cast(tuple_elt, type = cast_type)
    elts.append(tuple_elt)
  return tuple(elts)  
    