
import numpy as np 

from .. import prims 
from ..frontend import typed_macro 
from ..ndtypes import ScalarT, ArrayT, make_array_type
from ..syntax import PrimCall, Call,  OuterMap, Map
from ..syntax.helpers import make_closure 

def _get_vdot_fn(a, b):
  from reductions  import vdot  
  from ..frontend import ast_conversion 
  vdot_untyped = ast_conversion.translate_function_value(vdot)
  from ..type_inference import specialize
  vec_type_a = make_array_type(a.type.elt_type, 1) 
  vec_type_b = make_array_type(b.type.elt_type, 1)
  result_scalar_type = a.type.elt_type.combine(b.type.elt_type)
 
  vdot_typed = specialize(vdot_untyped, [vec_type_a, vec_type_b] )

  assert vdot_typed.return_type == result_scalar_type, \
    "Expected return type %s but got %s from vdot" % (result_scalar_type, vdot_typed.return_type)
  return vdot_typed 

@typed_macro
def dot(a,b):
  if isinstance(a.type, ScalarT):
    return PrimCall(prims.multiply, [a, b], type = b.type.combine(a.type))
  elif isinstance(b.type, ScalarT):
    return PrimCall(prims.multiply, [a, b], type = a.type.combine(b.type))
  
  assert isinstance(a.type, ArrayT), "Expected %s to be array but got %s" % (a, a.type)
  assert isinstance(a.type, ArrayT), "Expected %s to be array but got %s" % (b, b.type)
      
      
  if a.type.rank == 1 and b.type.rank == 1:
    vdot = _get_vdot_fn(a,b)
    return Call(fn = vdot, args = [a, b], type = vdot.return_type)

  elif a.type.rank == 1:
    vdot = _get_vdot_fn(a,b)
    
    assert b.type.rank == 2, "Don't know how to multiply %s and %s" % (a.type, b.type)
    vdot_col = make_closure(vdot, (a,))
    result_vec_type = make_array_type(vdot.return_type, 1)
    return Map(fn = vdot_col, args = (b,), axis = 1, type = result_vec_type)
        
  elif b.type.rank == 1:
    assert a.type.rank == 2, "Don't know how to multiply %s and %s" % (a.type, b.type)
    vdot = _get_vdot_fn(b,a)
    vdot_row = make_closure(vdot, (b,))
    result_vec_type = make_array_type(vdot.return_type, 1)
    return Map(fn = vdot_row, args = (a,), axis = 0, type = result_vec_type)
  else:  
    assert a.type.rank == 2 and b.type.rank == 2, \
        "Don't know how to multiply %s and %s" % (a.type, b.type)
    vdot = _get_vdot_fn(a,b)
    result_matrix_type = make_array_type(vdot.return_type, 2)
    return OuterMap(fn = vdot, args = (a, b), axis = (0,1), 
                    type = result_matrix_type)
      
@typed_macro 
def norm(x, ord=None):
  if ord is None:
    ord = 2
  if isinstance(x.type, ScalarT):
    return x
  assert isinstance(x.type, ArrayT), "Argument to 'norm' must be array, got %s : %s" % (x, x.type)
  assert x.type.rank == 1, "Norm currently only supported for vectors, not for %s : %s" (x, x.type)
  if ord == 0:
    return (x != 0).sum()
  elif ord == 1:
    return abs(x).sum()
  elif ord == 2:
    # not using 'conj' so won't yet work for complex numbers 
    return np.sqrt((x**2).sum())
  else:
    return (abs(x)**ord).sum() ** (1.0 / ord)

  
