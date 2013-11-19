from ..ndtypes import ArrayT 
from transform import Transform


class PermuteReductions(Transform):
  """
  When we have a reduction of array values, such as:
     Reduce(combine = Map(f), X, axis = 0) 
  it can be more efficient to interchange the Map and Reduce:
     Map(combine = f, X, axis = 1)
  """
  
  def transform_Reduce(self, expr):
    if self.is_none(expr.axis):
      return expr 
    if not isinstance(expr.type, ArrayT) or expr.type.rank != 2:
      return expr 
    
    fn = self.get_fn(expr.fn)
    fn_closure_args = self.closure_elts(expr.fn)
    
    combine = self.get_fn(expr.combine)
    combine_closure_args = self.closure_elts(expr.closure)
    
    