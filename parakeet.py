import types

# given the source of a function
# which can't refer to any global variables
# create a Python AST, convert it to Parakeet's
# untyped representation
def translate_function_from_source(s):
  return None


def run_compiled(fn, args):
  arg_types = map(types.type_of_value, args)
  specialized = specialize(fn, types)
  llvm_fn = llvm_compile(specialized)
  llvm_args = map(to_llvm_value, args)
  llvm_result = llvm_run(llvm_fn, llvm_args)
  return from_llvm_value(llvm_result)

