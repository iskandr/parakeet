
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


@jit 
def diff(x):
  """
  TODO:
    - axis selection
    - preserve size by filling with zeros
    - allow n'th differences by recursion
  """
  return x[1:] - x[:-1]


@jit 
def fill(x, v):
  for i in range(len(x)):
    x[i] = v 
 
