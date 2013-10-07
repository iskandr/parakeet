
import numpy as np 

from .. import prims 
from ..frontend import jit, macro, typed_macro 
from ..ndtypes import ScalarT, ArrayT, make_array_type
from ..syntax import Select, DelayUntilTyped, PrimCall, Call,  OuterMap

@typed_macro
def dot(a,b):
  if isinstance(a.type, ScalarT):
    return PrimCall(prims.multiply, [a, b], type = b.type.combine(a.type))
  elif isinstance(b.type, ScalarT):
    return PrimCall(prims.multiply, [a, b], type = a.type.combine(b.type))
  else:
    assert isinstance(a.type, ArrayT), \
      "Expected %s to be array but got %s" % (a, a.type)
    assert isinstance(a.type, ArrayT), \
      "Expected %s to be array but got %s" % (b, b.type)
      
    from numpy_reductions import vdot  
    from ..frontend import ast_conversion 
    vdot_untyped = ast_conversion.translate_function_value(vdot)
    from ..type_inference import specialize 
    vdot_typed = specialize(vdot_untyped, 
                            [make_array_type(a.type.elt_type, 1), 
                             make_array_type(b.type.elt_type, 1)])
    result_type = a.type.elt_type.combine(b.type.elt_type)
    assert vdot_typed.return_type == result_type, \
      "Expected return type %s but got %s from vdot" % (result_type, vdot_typed.return_type)
      
    if a.type.rank == 1 and b.type.rank == 1:
      return Call(fn = vdot_untyped, args = [a, b], type = result_type)

    if a.type.rank == 1:
      assert b.type.rank == 2, "Don't know how to multiply %s and %s" % (a.type, b.type)
      assert False, "Dot product between vector and matrix not yet implemented"
        
    elif b.type.rank == 1:
      assert a.type.rank == 1, "Don't know how to multiply %s and %s" % (a.type, b.type)
      assert False, "Dot product between matrix and vector not yet implemented"
    assert a.type.rank == 2 and b.type.rank == 2, \
        "Don't know how to multiply %s and %s" % (a.type, b.type)
      
    return OuterMap(fn = vdot_typed, args = (a, b), axis = (0,1), 
                    type = make_array_type(result_type, 2))
      
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

  