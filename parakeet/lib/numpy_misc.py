
from .. import prims 
from ..frontend import jit, macro
from ..ndtypes import ScalarT, ArrayT 
from ..syntax import Select, DelayUntilTyped, PrimCall, Call,  OuterMap

from numpy_array import transpose
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
      _vdot = ast_conversion.translate_function_value(vdot)
      
      result_type = X.type.elt_type.combine(Y.type.elt_type)
      if X.type.rank == 1 and Y.type.rank == 1:
        return Call(fn = _vdot, args = [X, Y], type = result_type)

      if X.type.rank == 1:
        assert Y.type.rank == 2, "Don't know how to multiply %s and %s" % (X.type, Y.type)

        assert False, "Dot product between vector and matrix not yet implemented"
        
      elif Y.type.rank == 1:
        assert X.type.rank == 1, "Don't know how to multiply %s and %s" % (X.type, Y.type)
        assert False, "Dot product between matrix and vector not yet implemented"
        
      assert X.type.rank == 2 and Y.type.rank == 2, \
        "Don't know how to multiply %s and %s" % (X.type, Y.type)
      return OuterMap(fn = _vdot, args = (X, Y), axis = 0, type = result_type)
  return DelayUntilTyped(  [X,  transpose.transform([Y])], typed_dot)
    