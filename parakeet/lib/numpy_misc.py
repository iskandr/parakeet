
from .. import prims 
from ..frontend import jit, macro, typed_macro 
from ..ndtypes import ScalarT, ArrayT, make_array_type
from ..syntax import Select, DelayUntilTyped, PrimCall, Call,  OuterMap



 
@macro
def _select(cond, trueval, falseval):
  return Select(cond, trueval, falseval)

#@macro  
def where(cond, trueval, falseval):
  #return Select(cond, trueval, falseval )
  return map(_select, cond, trueval, falseval)
