from .. syntax.helpers import unwrap_constant 
from transform import Transform 


class MapsToParFor(Transform):
  """
  Take all the adverbs whose parameterizing functions assume they 
  get fed slices of input data and turn them into version which take explicit
  input indices
  """
  _indexed_fn_cache = {}
  
  def mk_indexed_fn(fn,  new_closure_vals, k):
    """
    Take a function whose last k values are slices through input data 
    and transform it into a function which explicitly extracts its arguments
    """  
    pass 
  
  def transform_AllPairs(self, expr):
    axis = unwrap_constant(expr.axis)
    dimsizes = [self.shape(arg, axis) for arg in expr.args]
    