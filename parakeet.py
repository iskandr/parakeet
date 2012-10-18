import interp 
import ast_conversion
import type_inference
import type_conv 
import llvm_backend 


def expect(fn, args, expected):
  """
  Helper function used for testing, assert that Parakeet evaluates given code to
  the correct result
  """
  untyped  = ast_conversion.translate_function_value(fn)
  # should eventually roll this up into something cleaner, since 
  # top-level functions are really acting like closures over their
  # global dependencies 
  global_args = [fn.func_globals[n] for n in untyped.nonlocals]
  all_args = global_args + list(args)
  untyped_result = interp.eval_fn(untyped, *all_args) 
  assert untyped_result == expected, "Expected %s but got %s" % (expected, untyped_result)

  input_types = map(type_conv.typeof, all_args)
  typed = type_inference.specialize(untyped, input_types)
  typed_result = interp.eval_fn(typed, *all_args)
  assert typed_result == expected, "Expected %s but got %s" % (expected, typed_result)

  compiled = llvm_backend.compile_fn(typed)
  llvm_result = compiled(*all_args)
  assert llvm_result == expected, "Expected %s but got %s" % (expected, llvm_result)

  
