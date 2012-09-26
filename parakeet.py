import interp 



"""
def run_compiled(fn, args):
  arg_types = map(parakeet_types.type_of_value, args)
  specialized = specialize(fn, types)
  llvm_fn = llvm_compile(specialized)
  llvm_args = map(to_llvm_value, args)
  llvm_result = llvm_run(llvm_fn, llvm_args)
  return from_llvm_value(llvm_result)

"""