from ..builder import build_fn 
from ..ndtypes import Int64, repeat_tuple
from ..syntax import ParFor, IndexReduce, IndexScan, IndexFilter 
from ..syntax.helpers import unwrap_constant, get_types 
from transform import Transform 



class IndexifyAdverbs(Transform):
  """
  Take all the adverbs whose parameterizing functions assume they 
  get fed slices of input data and turn them into version which take explicit
  input indices
  """
  _indexed_fn_cache = {}
  def indexify_fn(self, fn, array_args, n_indices):
    """
    Take a function whose last k values are slices through input data 
    and transform it into a function which explicitly extracts its arguments
    """  
    array_arg_types = tuple(get_types(array_args))
    # do I need fn.version *and* fn.copied_by? 
    key = (fn.name, fn.copied_by, fn.version, array_arg_types, n_indices)
    if key in self._indexed_fn_cache:
      return self._indexed_fn_cache[key]
    input_vars = self.input_vars(fn)
    k = len(array_arg_types)
    closure_args = input_vars[:-k]
    slice_args = input_vars[-k:]

    array_arg_vars = tuple(self.fresh_var(t, prefix="array_arg%d" % i)
                           for i,t in enumerate(array_arg_types))
    new_closure_args = tuple(closure_args) + array_arg_vars
    index_arg = Int64 if n_indices == 1 else repeat_tuple(Int64, n_indices) 
    new_fn = build_fn()
    
  
  def transform_Map(self, expr):
    args = self.transform_expr_list(expr.args)
    axis = unwrap_constant(expr.axis)
    # recursively descend down the function bodies to pull together nested ParFors 
    return ParFor()
  def transform_OuterMap(self, expr):
    args = self.transform_expr_list(expr.args)
    axis = unwrap_constant(expr.axis)
    dimsizes = [self.shape(arg, axis) for arg in args]
    # recursively descend down the function bodies to pull together nested ParFors 
    return ParFor()
  
  def transform_IndexMap(self, expr):
    # recursively descend down the function bodies to pull together nested ParFors 
    return ParFor()
  
  def transform_Reduce(self, expr):
    return IndexReduce()
  
  def transform_Scan(self, expr):
    return IndexScan()
  
  def transform_Filter(self, expr):
    return IndexFilter(self, expr)