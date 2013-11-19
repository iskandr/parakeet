from ..builder import build_fn
from ..ndtypes import ArrayT, lower_rank
from ..prims import Prim  
from ..syntax import Return, Map 
from ..syntax.helpers import is_identity_fn, unwrap_constant
from transform import Transform

def get_nested_map(fn):
  if len(fn.body) != 1:
    return None 
  stmt = fn.body[0]
  if stmt.__class__ is not Return:
    return None 
  if stmt.value.__class__ is not Map:
    return None 
  return stmt.value 
     

class PermuteReductions(Transform):
  """
  When we have a reduction of array values, such as:
     Reduce(combine = Map(f), X, axis = 0) 
  it can be more efficient to interchange the Map and Reduce:
     Map(combine = f, X, axis = 1)
  """
  
  
  def transform_Reduce(self, expr):
    return expr 
  
  def transform_Scan(self, expr):
    if len(expr.args) > 1:
      return expr 
    
    axis = unwrap_constant(expr.axis)
    if axis is None or not isinstance(axis, (int,long)) or axis > 1 or axis < 0:
      return expr
     
    if not isinstance(expr.type, ArrayT) or expr.type.rank != 2:
      return expr
    
    fn = self.get_fn(expr.fn)
    fn_closure_args = self.closure_elts(expr.fn)
    if len(fn_closure_args) > 0:
      return expr 
    
    combine = self.get_fn(expr.combine)
    combine_closure_args = self.closure_elts(expr.closure)
    if len(combine_closure_args) > 0:
      return expr 
    
    if is_identity_fn(fn):
      nested_map = get_nested_map(combine)
    
    if not isinstance(nested_map.fn, Prim):
      return expr  
    
    arg_t = expr.args[0].type
    elt_t = lower_rank(arg_t, 1)
    new_nested_fn = None 
    return Map(fn = new_nested_fn,  
               args = expr.args, 
               axis = 1 - axis, 
               type = expr.type)
      
       
  
    