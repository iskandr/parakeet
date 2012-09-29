import interp 
import ast_conversion
import ptype
import type_analysis

def expect(fn, args, expected):
  """
  Helper function used for testing, assert that Parakeet evaluates given code to
  the correct result
  """
  untyped_result = interp.run(fn, args, type_specialization = False)
  assert untyped_result == expected, "Expected %s but got %s" % (expected, untyped_result)
 
  typed_result = interp.run(fn, args, type_specialization = True)
  assert typed_result == expected, "Expected %s but got %s" % (expected, typed_result)
  
"""
def run_compiled(fn, args):
  arg_types = map(parakeet_types.type_of_value, args)
  specialized = specialize(fn, types)
  llvm_fn = llvm_compile(specialized)
  llvm_args = map(to_llvm_value, args)
  llvm_result = llvm_run(llvm_fn, llvm_args)
  return from_llvm_value(llvm_result)

"""