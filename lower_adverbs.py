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

  def adverb_prelude(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis) 
    if axis is None:
      args = self.flatten_array_args(args)
      axis = 0
    return fn, args, axis
  

  
  
  def transform_Map(self, expr):
    fn, args, axis = self.adverb_prelude(expr)
      
    if all( arg.type.rank == 0 for arg in args ):
      return syntax.Invoke(expr.fn, args, type = expr.type)
    
      
    # TODO: Should make sure that all the shapes conform here, 
    # but we don't yet have anything like assertions or error handling
    max_arg = adverb_helpers.max_rank_arg(args)
    niters = self.shape(max_arg, axis)
    
    i, i_after, merge = self.loop_counter("i")
    
    cond = self.lt(i, niters)
    elt_t = expr.type.elt_type
    array_result = self.alloc_array(elt_t, niters)
    self.blocks.push()
    nested_args = [self.index_along_axis(arg, axis, i) for arg in args]

    call_result = self.invoke(fn, nested_args)    
    output_idx = syntax.Index(array_result, i, type = call_result.type)
    self.assign(output_idx, call_result)
    self.assign(i_after, self.add(i, syntax_helpers.one_i64))

    body = self.blocks.pop()
    self.blocks += syntax.While(cond, body, merge)
    return array_result 
  
  
  def transform_Reduce(self, expr):
    fn, args, axis = self.adverb_prelude(expr)
    # For now we only work with a single array 
    # and ignore init parameters 
    assert len(args) == 1, \
      "Reduce currently can't handle more than one input, given: %s" % (args,)
    x = args[0]
    n = self.shape(x, axis)
    x0 = self.index_along_axis(x, axis, 0, "first_elt")
    x1 = self.index_along_axis(x, axis, 0, "second_elt")
    init = self.invoke(fn, [x0, x1])
    
    i, i_after, merge = self.loop_counter("i", syntax_helpers.const(2))
    acc = self.fresh_var(init.type, "acc")
    
    cond = self.lt(i, n)
    self.blocks.push()
    curr_elt = self.index_along_axis(x, axis, i, "curr_elt")
    new_acc = self.invoke(fn, [acc, curr_elt])
    assert new_acc.type == acc.type, \
      "Inconsistent accumulator type in reduction: %s != %s" % \
      (acc.type, new_acc.type)
    merge[acc.name] = (init, new_acc)
    self.assign(i_after, self.add(i, syntax_helpers.one_i64))
    body = self.blocks.pop()
    self.blocks += syntax.While(cond, body, merge)
    return acc   
      
  
  def transform_AllPairs(self, expr):

    fn, args, axis = self.adverb_prelude(expr)
    
    if all( arg.type.rank == 0 for arg in args ):
      return syntax.Invoke(expr.fn, args, type = expr.type)
    
    x, y = args 
    nx = self.shape(x, axis)
    ny = self.shape(y, axis)
    
    elt_t = expr.type.elt_type
    array_result = self.alloc_array(elt_t, (nx, ny))
    
    i, i_after, merge_i = self.loop_counter("i")
    cond_i = self.lt(i, nx)
    self.blocks.push()
    
    j, j_after, merge_j = self.loop_counter("j")
    cond_j = self.lt(j, ny)
    self.blocks.push()
    
    nested_args = [self.index_along_axis(x, axis, i), 
                   self.index_along_axis(y, axis, j)]
    invoke = self.invoke(fn, nested_args)
    indices = self.tuple([i, j], "indices")
    output_idx = syntax.Index(array_result, indices, type = invoke.type)
    self.assign(output_idx, invoke)
    
    self.assign(j_after, self.add(j, syntax_helpers.one_i64))
    inner_body = self.blocks.pop()
    self.blocks += syntax.While(cond_j, inner_body, merge_j )
    
    self.assign(i_after, self.add(i, syntax_helpers.one_i64))

    outer_body = self.blocks.pop()
    self.blocks += syntax.While(cond_i, outer_body, merge_i)
    return array_result 
      
  
def lower_adverbs(fn):
  return transform.cached_apply(LowerAdverbs, fn)
