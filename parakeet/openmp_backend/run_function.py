from .. import config 

from ..c_backend.prepare_args import prepare_args  
from ..transforms.pipeline import after_indexify, final_loop_optimizations, flatten  
from ..value_specialization import specialize


from multicore_compiler import MulticoreCompiler 

_cache = {}
def run(fn, args):
  args = prepare_args(args, fn.input_types)
  fn = after_indexify(fn)
  fn = final_loop_optimizations.apply(fn)
  if config.value_specialization:
    fn = specialize(fn, python_values = args)
  compiled_fn = MulticoreCompiler().compile_entry(fn)
  return compiled_fn.c_fn(*args)
  