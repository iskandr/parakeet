import interp 
import ast_conversion
import specialization
import ptype
import llvm_backend 
import llvm_runtime


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

  input_types = map(ptype.type_of_value, all_args)
  typed = specialization.specialize(untyped, input_types)
  typed_result = interp.eval_fn(typed, *all_args)
  assert typed_result == expected, "Expected %s but got %s" % (expected, typed_result)

  llvm_fn = llvm_backend.compile_fn(typed)
  llvm_result = llvm_runtime.run(llvm_fn, all_args)
  assert llvm_result == expected, "Expected %s but got %s" % (expected, llvm_result)
  


"""
def run_compiled(fn, args):
  arg_types = map(parakeet_types.type_of_value, args)
  specialized = specialize(fn, types)
  llvm_fn = llvm_compile(specialized)
  llvm_args = map(to_llvm_value, args)
  llvm_result = llvm_run(llvm_fn, llvm_args)
  return from_llvm_value(llvm_result)

"""