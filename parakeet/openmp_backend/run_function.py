from .. import config, c_backend 

from ..c_backend.prepare_args import prepare_args  
from ..transforms.pipeline import flatten
from ..transforms.stride_specialization import specialize

from compiler import MulticoreModuleCompiler 

def run(fn, args):
  args = prepare_args(args, fn.input_types)
  fn = flatten(fn)
  print "Running in OpenMP", fn 
  if config.stride_specialization:
    fn = specialize(fn, python_values = args)
  print "After specialization", fn 
      
  compiled_fn = c_backend.compile_entry(fn, compiler_class = MulticoreModuleCompiler)

  assert len(args) == len(fn.input_types)
  result = compiled_fn.c_fn(*args)
  return result