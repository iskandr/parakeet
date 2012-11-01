
import ast_conversion 
import syntax 
import type_inference 
import type_conv
import llvm_backend 

def specialize_and_compile(fn, args):
  if isinstance(fn, syntax.Fn):
    untyped = fn 
  else:
    # translate from the Python AST to Parakeet's untyped format 
    untyped  = ast_conversion.translate_function_value(fn)
  all_args = untyped.python_nonlocals() + list(args)
  
  # get types of all inputs
  input_types = [type_conv.typeof(arg) for arg in all_args]
  
  # propagate types through function representation and all
  # other functions it calls 
  typed = type_inference.specialize(untyped, input_types)
  
  # compile to native code 
  compiled = llvm_backend.compile_fn(typed)

  return untyped, typed, all_args, compiled 
  
def run(fn, args):
  _, _, all_args, compiled = specialize_and_compile(fn, args)
  return compiled(*all_args)

import adverb_helpers
import adverbs

def create_adverb_hook(adverb_class, default_args = ['x'], default_axis = None):
  def create_wrapper(fundef, **kwds):
    if not isinstance(fundef, syntax.Fn):
      fundef = ast_conversion.translate_function_value(fundef)
    assert len(fundef.args.defaults) == 0
    arg_names = ['fn'] + list(fundef.args.positional)
    return adverb_helpers.untyped_wrapper(adverb_class, arg_names, **kwds)
  def python_hook(fn, *args, **kwds):
    wrapper = create_wrapper(fn, **kwds)
    return run(wrapper, [fn] + list(args))
  # for now we register with the default number of args since our wrappers
  # don't yet support unpacking a variable number of args
  default_wrapper_args = ['fn'] + default_args
  
  default_wrapper = adverb_helpers.untyped_wrapper(adverb_class, default_wrapper_args, axis=default_axis)
  adverb_helpers.register_adverb(python_hook, default_wrapper)
  return python_hook


each = create_adverb_hook(adverbs.Map)
allpairs = create_adverb_hook(adverbs.AllPairs, default_args = ['x','y'], default_axis = 0)
seq_reduce = create_adverb_hook(adverbs.Reduce, default_args = ['acc', 'x'])
seq_scan = create_adverb_hook(adverbs.Scan, default_args = ['acc', 'x'])
