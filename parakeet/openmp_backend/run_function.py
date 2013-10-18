from .. import config 

from ..c_backend.prepare_args import prepare_args  
from ..transforms.pipeline import flatten, indexify 
from ..transforms.stride_specialization import specialize

from compiler import MulticoreCompiler 

def run(fn, args):
  args = prepare_args(args, fn.input_types)
  print "Before indexify", fn 
  fn = indexify(fn)
  print "Before flatten", fn 
  fn = flatten(fn)
   
  if config.stride_specialization:
    fn = specialize(fn, python_values = args)
      
  compiled_fn = MulticoreCompiler().compile_entry(fn)
 
  assert len(args) == len(fn.input_types)
  result = compiled_fn.c_fn(*args)
  return result