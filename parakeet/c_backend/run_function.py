from prepare_args import prepare_args
from ..transforms.pipeline  import loopify, final_loop_optimizations  
from ..value_specialization import specialize
from ..config import value_specialization
from pymodule_compiler import PyModuleCompiler 



_cache = {}
def run(fn, args):
  args = prepare_args(args, fn.input_types)
  
  transformed_fn = loopify.apply(fn)
  transformed_fn = final_loop_optimizations.apply(transformed_fn)
  if value_specialization: 
    transformed_fn = specialize(transformed_fn, args)
  
  key = transformed_fn.cache_key
  if key in _cache:
    c_fn = _cache[key]
  else:
    compiled_fn = PyModuleCompiler().compile_entry(transformed_fn)
    c_fn = compiled_fn.c_fn 
    _cache[key] = c_fn 
  return c_fn(*args)
  