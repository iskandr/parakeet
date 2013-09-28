
from .. import prims 
from ..frontend import jit, macro
from ..ndtypes import ScalarT, ArrayT, make_array_type
from ..syntax import Select, DelayUntilTyped, PrimCall, Call,  OuterMap


from numpy_reductions import vdot
 
@macro
def _select(cond, trueval, falseval):
  return Select(cond, trueval, falseval)

#@macro  
def where(cond, trueval, falseval):
  #return Select(cond, trueval, falseval )
  return map(_select, cond, trueval, falseval)

@macro
def dot(X,Y):
  def typed_dot(X, Y):
    if isinstance(X.type, ScalarT):
      return PrimCall(prims.multiply, [X, Y], type = Y.type.combine(X.type))
    elif isinstance(Y.type, ScalarT):
      return PrimCall(prims.multiply, [X, Y], type = X.type.combine(Y.type))
    else:
      assert isinstance(X.type, ArrayT), \
        "Expected %s to be array but got %s" % (X, X.type)
      assert isinstance(X.type, ArrayT), \
        "Expected %s to be array but got %s" % (Y, Y.type)
      
      from ..frontend import ast_conversion 
      vdot_untyped = ast_conversion.translate_function_value(vdot)
      from ..type_inference import specialize 
      vdot_typed = specialize(vdot_untyped, 
                              [make_array_type(X.type.elt_type, 1), 
                               make_array_type(Y.type.elt_type, 1)])
      result_type = X.type.elt_type.combine(Y.type.elt_type)
      assert vdot_typed.return_type == result_type, \
        "Expected return type %s but got %s from vdot" % (result_type, vdot_typed.return_type)
      
      if X.type.rank == 1 and Y.type.rank == 1:
        return Call(fn = vdot_untyped, args = [X, Y], type = result_type)

      if X.type.rank == 1:
        assert Y.type.rank == 2, "Don't know how to multiply %s and %s" % (X.type, Y.type)

        assert False, "Dot product between vector and matrix not yet implemented"
        
      elif Y.type.rank == 1:
        assert X.type.rank == 1, "Don't know how to multiply %s and %s" % (X.type, Y.type)
        assert False, "Dot product between matrix and vector not yet implemented"
        
      assert X.type.rank == 2 and Y.type.rank == 2, \
        "Don't know how to multiply %s and %s" % (X.type, Y.type)
      
      return OuterMap(fn = vdot_typed, args = (X, Y), axis = (0,1), 
                      type = make_array_type(result_type, 2))
      
  return DelayUntilTyped(  [X, Y], typed_dot)
    