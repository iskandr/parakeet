
import ast_conversion
import type_inference
import type_conv 
import llvm_backend 



def specialize_and_compile(fn, args):
  # translate from the Python AST to Parakeet's untyped format 
  untyped  = ast_conversion.translate_function_value(fn)
  
  # should eventually roll this up into something cleaner, since 
  # top-level functions are really acting like closures over their
  # global dependencies 
  global_args = [fn.func_globals[n] for n in untyped.nonlocals]
  all_args = global_args + list(args)
  
  # get types of all inputs
  input_types = map(type_conv.typeof, all_args)
  
  # propagate types through function representation and all
  # other functions it calls 
  typed = type_inference.specialize(untyped, input_types)
  
  # compile to native code 
  compiled = llvm_backend.compile_fn(typed)
  return untyped, typed, all_args, compiled 
  


def run(fn, args):
  _, _, all_args, compiled = specialize_and_compile(fn, args)
  return compiled(*all_args)


import adverbs 

def map(fn, *args):
  untyped  = ast_conversion.translate_function_value(fn)
  wrapper = adverbs.untyped_wrapper(adverbs.Map, untyped)
  return run(wrapper, args)

from prelude import * 