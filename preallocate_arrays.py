import names 
from syntax import Assign, Index, Slice, Var, Return 
from syntax_helpers import none, slice_none, zero_i64 
from transform import MemoizedTransform 
from clone_function import CloneFunction
from core_types import NoneT 
from array_type import ArrayT 
import shape_inference
from adverb_helpers import max_rank
import shape 
class PreallocateOutputs(MemoizedTransform):
  """
  Transform a function so that instead of returning a value
  it writes to an additional output parameter
  """
  def pre_apply(self, fn):
    # create a fresh copy 
    fn = CloneFunction().apply(fn)
    output_name = names.fresh("output")
    output_t = fn.return_type
    output_var = Var(output_name, type=output_t)
    self.output_type = output_t
    if output_t.__class__ is ArrayT and output_t.rank > 0: 
      idx = self.tuple([slice_none] * output_t.rank)
      self.output_lhs_index = Index(output_var, idx)
      #self.output_value = self.output_lhs_index 
    else:
      self.output_lhs_index = Index(output_var, none)
      #self.output_value = Index(output_var, zero_i64)
    
    fn.return_type = NoneT  
    fn.arg_names = fn.arg_names + (output_name,)
    fn.input_types = fn.input_type + (output_t,)
    self.return_none = Return(none)
    return fn 

  def transform_Return(self, stmt):
    self.assign(self.output_lhs_index, stmt.value)
    return self.return_none

def preallocate_outputs(fn):
  return PreallocateOutputs().apply(fn)

class PreallocateArrays(MemoizedTransform):
  def visit_TypedFn(self, expr):
    return preallocate_arrays(expr)
    
  def map_shape(self, maybe_clos, args, axis):
    fn = self.get_fn(maybe_clos)
    closure_elts = self.closure_elts(maybe_clos)
    nested_shape = shape_inference.call_shape_expr(fn)
    max_rank = -1
    biggest_arg = None 
    for arg in args:
      r = self.rank(arg)
      if r > max_rank:
        max_rank = r 
        biggest_arg = arg 
    outer_axis = shape.
    
    
  def visit_Map(self, expr):
    
    # transform the function to take an additional output parameter 
    fn = preallocate_outputs(expr.fn)
     

def preallocate_arrays(fn):
  return PreallocateArrays().apply(fn)