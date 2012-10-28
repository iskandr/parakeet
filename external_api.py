import adverb_helpers
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

  all_args = untyped.get_closure_args() + list(args)
  # get types of all inputs
  input_types = [type_conv.typeof(arg) for arg in  all_args]
  
  # propagate types through function representation and all
  # other functions it calls 
  typed = type_inference.specialize(untyped, input_types)
  
  # compile to native code 
  compiled = llvm_backend.compile_fn(typed)

  return untyped, typed, all_args, compiled 
  

def run(fn, args):
  _, _, all_args, compiled = specialize_and_compile(fn, args)
  return compiled(*all_args)


def each(fn, *args):
  """
  Apply fn to each element of the input arguments
  """
  untyped  = ast_conversion.translate_function_value(fn)
  wrapper = adverb_helpers.untyped_map_wrapper(untyped)
  return run(wrapper, args)

def allpairs(fn, x, y, axis = None):
  untyped = ast_conversion.translate_function_value(fn)
  wrapper = adverb_helpers.untyped_allpairs_wrapper(untyped, axis)
  return run(wrapper, [x, y])

def seq_reduce(fn, *args):
  """
  Sequential reductions lack a 'combine' operator and thus
  can't be split apart into parallel sub-tasks
  """
  untyped = ast_conversion.translate_function_value(fn)
  wrapper = adverb_helpers.untyped_reduce_wrapper(untyped)
  return run(wrapper, args)

def seq_scan(fn, *args):
  """
  Prefix scan of the given function over elements of the inputs arguments. 
  A sequential scan lacks a 'combine' operator and thus can't be parallelized
  """ 
  untyped = ast_conversion.translate_function_value(fn)
  wrapper = adverb_helpers.untyped_scan_wrapper(untyped)
  return run(wrapper, args)
