import transform
import syntax
import core_types
import array_type
import adverb_helpers 
import syntax_helpers
import type_inference

class LowerAdverbs(transform.Transform):
  
  def flatten_array_arg(self, x):
    if isinstance(x.type, core_types.ScalarT):
      return x
    else:
      
      assert isinstance(x.type, array_type.ArrayT), \
        "Arguments to adverbs must be scalars or arrays, got %s" % x
        
      if x.type.rank <= 1:
        return x
      else:        
        #print "RAVEL ARG", x, x.type 
        ravel_t = array_type.make_array_type(x.type.elt_type, 1)
        # TODO: Replace this dummy string with an actual ravel primitive 
        return self.assign_temp(syntax.Ravel(x, type = ravel_t), "ravel")
  
  def flatten_array_args(self, xs):
    return map(self.flatten_array_arg, xs)

  def transform_Map(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    if all( [arg.type.rank == 0 for arg in args] ):
      return syntax.Invoke(expr.fn, args, type = expr.type)
    
    axis = expr.axis 
    if axis is None:
      args = self.flatten_array_args(args)
      axis = 0
      
    # TODO: Should make sure that all the shapes conform here, 
    # but we don't yet have anything like assertions or error handling
    max_arg = adverb_helpers.max_rank_arg(args)
    niters = self.shape(max_arg, axis)
    
    # UGH generating code into SSA form is annoying 
    counter_before = self.zero_i64("i_before")
    counter = self.fresh_i64("i_loop")
    counter_after = self.fresh_i64("i_after")
    
    merge = { counter.name : (counter_before, counter_after) }
    
    
    
    cond = self.lt(counter, niters)
    elt_t = expr.type.elt_type
    array_result = self.alloc_array(elt_t, niters)
    self.blocks.push()
    nested_args = [self.index(arg, counter) for arg in args]
    closure_t = fn.type
    nested_arg_types = syntax_helpers.get_types(nested_args)
    call_result_t = type_inference.invoke_result_type(closure_t, nested_arg_types)
    call = syntax.Invoke(fn, nested_args, type = call_result_t)
    call_result = self.assign_temp(call, "call_result")
    output_idx = syntax.Index(array_result, counter, type = call_result.type)
    self.assign(output_idx, call_result)
    self.assign(counter_after, self.add(counter, syntax_helpers.one_i64))
  
    body = self.blocks.pop()
    self.blocks += syntax.While(cond, body, merge)
    return array_result 
  
  def transform_AllPairs(self, expr):
    fn = self.transform_expr(expr.fn)

    args = self.transform_expr_list(expr.args)
    if all( [arg.type.rank == 0 for arg in args] ):
      return syntax.Invoke(expr.fn, args, type = expr.type)
    
    x, y = args 
    
    axis = expr.axis 
    if axis is None:
      args = self.flatten_array_args(args)
      axis = 0
      
    nx = self.shape(x, axis)
    ny = self.shape(y, axis)
    
    i_before = self.zero_i64("i_before")
    i = self.fresh_i64("i_loop")
    i_after = self.fresh_i64("i_after")
    
    merge_i = { i.name : (i_before, i_after) }
    
    j_before = self.zero_i64("j_before")
    j = self.fresh_i64("j")
    j_after = self.fresh_i64("i_after")
    
    merge_j = { j.name : (j_before, j_after) }
    
    cond_i = self.lt(i, nx)
    cond_j = self.lt(j, ny)
    
    elt_t = expr.type.elt_type
    array_result = self.alloc_array(elt_t, (nx, ny))
    self.blocks.push()
    nested_args = [self.index(arg, (i, j)) for arg in args]
    closure_t = fn.type
    nested_arg_types = syntax_helpers.get_types(nested_args)
    call_result_t = type_inference.invoke_result_type(closure_t, nested_arg_types)
    call = syntax.Invoke(fn, nested_args, type = call_result_t)
    call_result = self.assign_temp(call, "call_result")
    indices = self.tuple([i, j], "indices")
    output_idx = syntax.Index(array_result, indices, type = call_result.type)
    self.assign(output_idx, call_result)
    self.assign(i_after, self.add(i, syntax_helpers.one_i64))
    body = self.blocks.pop()
    self.blocks += syntax.While(cond_i, body, merge_i)
    return array_result 
      
  

def lower_adverbs(fn):
  return transform.cached_apply(LowerAdverbs, fn)
  