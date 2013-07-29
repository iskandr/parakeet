from ..syntax import Select 
from ..frontend import jit, macro

@macro
def _select(cond, trueval, falseval):
  return Select(cond, trueval, falseval)

#@macro  
def where(cond, trueval, falseval):
  #return Select(cond, trueval, falseval )
  return map(_select, cond, trueval, falseval)