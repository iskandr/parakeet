

def run(fn, args):
  from pipeline import lowering 
  lowered = lowering.apply(fn)
  
  if config.stride_specialization:
   lowered = stride_specialization.specialize(lowered, arg_values)

  # compile to native code
  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(lowered)
  compiled_fn_wrapper = CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  return untyped, typed, compiled_fn_wrapper, arg_values
