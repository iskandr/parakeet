
from .. import prims 
from ..frontend import jit, macro
from ..ndtypes import ScalarT, ArrayT 
from ..syntax import Select, DelayUntilTyped, PrimCall, Call
 
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
  def typed_dot(Xt, Yt):
    if isinstance(Xt.type, ScalarT):
      return PrimCall(prims.multiply, [Xt, Yt], type = Yt.type.combine(Xt.type))
    elif isinstance(Yt.type, ScalarT):
      return PrimCall(prims.multiply, [Xt, Yt], type = Xt.type.combine(Yt.type))
    else:
      assert isinstance(Xt.type, ArrayT), \
        "Expected %s to be array but got %s" % (Xt, Xt.type)
      assert isinstance(Xt.type, ArrayT), \
        "Expected %s to be array but got %s" % (Yt, Yt.type)
      assert False, "Dot product not yet implemented"
  return DelayUntilTyped((X,Y), typed_dot)
    