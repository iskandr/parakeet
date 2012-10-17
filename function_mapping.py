import ast_conversion 
import interp 
import function_registry 

def exec_untyped(untyped, fn_globals, args):
  global_args = [fn_globals[n] for n in untyped.nonlocals]
  all_args = global_args + list(args)
  return interp.eval_fn(untyped, *all_args) 

    
def remap(old_fn):
  """
  Given a python function value i.e. numpy.alen, 
  Take a python definition of its parakeet-friendly replacement
  and wrap it so that:
  1) when the reimplementation is called from outside parakeet
     it launches the parakeet runtime
  2) when either the original version of the python fn or its
     reimplementation are called from within parakeet it's 
     rerouted to the untyped AST of the reimplementation
  """
  def wrap(new_fn):
    # translates function and adds new_fn -> parakeet ast 
    # entry to function_regsitry
    untyped_ast = ast_conversion.translate_function_value(new_fn)
    def wrapped(*args):
      return  exec_untyped(untyped_ast, new_fn.func_globals, args)
    function_registry.register_python_fn(wrapped, untyped_ast)
    return wrapped
  return wrap 
    
    