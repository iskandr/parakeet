from transform import Transform 
import closure_type 
import function_registry 
from args import ActualArgs
import syntax_helpers
import syntax 

class SimplifyInvoke(Transform):
  def linearize_invoke(self, fn, old_args):
    
    new_fn = self.transform_expr(fn)
    fn_t = new_fn.type
    
    if isinstance(fn, syntax.TypedFn):
      assert isinstance(old_args, list)
      new_args = self.transform_expr_list(old_args)
      assert len(new_args) == len(new_fn.input_types)
      arg_types = syntax_helpers.get_types(new_args)
      assert all(t1 == t2 for (t1,t2) in zip(arg_types, fn.input_types))
      return fn, new_args, arg_types 
    elif isinstance(new_fn, syntax.Fn):
      untyped_fn = new_fn
      closure_args = []
      
    elif isinstance(fn_t, closure_type.ClosureT):
      untyped_fn = fn_t.fn
      if isinstance(untyped_fn, str):
        untyped_fn = function_registry.untyped_functions[untyped_fn]
      n_closure_args = len(fn_t.arg_types) 
      closure_args = \
        [self.closure_elt(new_fn, i) 
         for i in xrange(n_closure_args)]
    
    if isinstance(old_args, (list, tuple)):
      old_args = ActualArgs(old_args)
    old_args = old_args.prepend_positional(closure_args)
    new_args = old_args.transform(self.transform_expr)
    arg_types = new_args.transform(syntax_helpers.get_type)
    # Drop arguments that are assigned defaults, 
    # since we're assuming those are set in the body 
    # of the function 
    linear_args, extra = untyped_fn.args.linearize_values(new_args, 
                                                          tuple_elts_fn = self.tuple_elts, 
                                                          keyword_fn = lambda k, v: None)
    combined_args = [x for x in (linear_args + extra) if x] 

    return untyped_fn, combined_args, arg_types 
   
  def transform_Invoke(self, expr):
    untyped_fn, combined_args, types = \
      self.linearize_invoke(expr.closure, expr.args)
    
    import type_inference 
    typed_fundef = type_inference.specialize(untyped_fn, types)
    return syntax.Call(typed_fundef.name, combined_args, type = expr.type)