import numpy as np

from ..frontend import macro, jit 
from ..syntax import UntypedFn, Return, Cast, TypedFn, TypeValue, Array, Tuple, Expr 
from ..syntax.helpers import get_types  
from ..ndtypes import type_conv, TypeValueT, ArrayT, IntT, make_tuple_type, TupleT, ScalarT


def _get_type(dtype):
  """
  Defensively try to extract a scalar type from any wacky thing
  that might get passed in as a 'dtype' to an array constructor
  """
  if isinstance(dtype, macro):
    dtype = dtype.as_fn()
  
  while isinstance(dtype, jit):
    dtype = dtype.f 
    
  if isinstance(dtype, (np.dtype, type)):
    return type_conv.equiv_type(dtype)
  elif isinstance(dtype, str):
    return type_conv.equiv_type(np.dtype(dtype))
  elif isinstance(dtype, UntypedFn):
    if len(dtype.body) == 1:
      stmt = dtype.body[0]
      if stmt.__class__ is Return:
        expr = stmt.value 
        if expr.__class__ is Cast:
          return expr.type 
  elif isinstance(dtype, TypedFn):
    return dtype.return_type 
  elif isinstance(dtype, TypeValueT):
    return dtype.type
  elif isinstance(dtype, TypeValue):
    return dtype.type_value  
  elif isinstance(dtype, Expr):
    if isinstance(dtype.type, TypeValueT):
      return dtype.type.type 
    assert False, "Don't know how to turn %s : %s into Parakeet type" % (dtype, dtype.type)
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
    return Tuple(elts = (value,), type = make_tuple_type((value.type,)))
  assert False, "Can't make shape tuple from value %s : %s" % (value, value.type)  
    