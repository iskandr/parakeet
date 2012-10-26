import transform
import syntax
import core_types
import array_type
import adverb_helpers 
import syntax_helpers

class LowerAdverbs(transform.Transform):
  
  def flatten_array_arg(self, x):
    if isinstance(x.type, core_types.ScalarT):
      return x
    else:
      
      assert isinstance(x.type, array_type.ArrayT), \
        "Arguments to adverbs must be scalars or arrays, got %s" % x
        
      if x.type.rank == 1:
        return x
      else:        
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
    result = self.alloc_array(expr.type.elt_type, niters)
    self.blocks.push()
    nested_args = [self.index(arg, counter) for arg in args]
    call = syntax.Invoke(fn, nested_args, type = expr.type.elt_type)
    output_idx = syntax.Index(result, counter)
    self.blocks += [syntax.Assign(output_idx, call)]
    self.blocks += [syntax.Assign(counter_after, self.add(counter, syntax_helpers.one_i64))]
    body = self.blocks.pop()
    self.blocks += syntax.While(cond, body, merge, merge)
    return result 
    
  

def lower_adverbs(fn):
  return transform.cached_apply(LowerAdverbs, fn)
  