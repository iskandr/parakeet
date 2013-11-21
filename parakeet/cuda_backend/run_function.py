from ..c_backend.prepare_args import prepare_args 
from ..config import value_specialization 
from ..transforms.pipeline import (flatten, high_level_optimizations, after_indexify, 
                                   final_loop_optimizations) 
from ..value_specialization import specialize


from cuda_compiler import CudaCompiler 

def run(fn, args):
  args = prepare_args(args, fn.input_types)
  fn = after_indexify.apply(fn)
  # fn = flatten(fn)
  fn = final_loop_optimizations.apply(fn)
  if value_specialization:
    fn = specialize(fn, python_values = args)
  compiled_fn = CudaCompiler().compile_entry(fn)
  assert len(args) == len(fn.input_types)
  result = compiled_fn.c_fn(*args)
  return result
  