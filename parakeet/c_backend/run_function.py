from prepare_args import prepare_args
from ..transforms.pipeline  import loopify, final_loop_optimizations, flatten  
from ..transforms.stride_specialization import specialize
from ..config import stride_specialization
from pymodule_compiler import PyModuleCompiler 

def run(fn, args):
  args = prepare_args(args, fn.input_types)
  fn = loopify.apply(fn)
  # TODO: finish debuggin flattening 
  # fn = flatten(fn)
  fn = final_loop_optimizations.apply(fn)

  if stride_specialization:
    fn = specialize(fn, python_values = args)
  compiled_fn = PyModuleCompiler().compile_entry(fn)
  assert len(args) == len(fn.input_types)
  result = compiled_fn.c_fn(*args)
  return result