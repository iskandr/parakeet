from prepare_args import prepare_args
from ..transforms.pipeline  import loopify, final_loop_optimizations  
from ..value_specialization import specialize
from ..config import value_specialization
from pymodule_compiler import PyModuleCompiler 

_transform_cache = {} 
_compile_cache = {}

def run(fn, args):
  args = prepare_args(args, fn.input_types)
  
  key = fn.cache_key
  if key in _transform_cache:
    transformed_fn = _transform_cache[key]
  else:
    transformed_fn = loopify.apply(fn)
    transformed_fn = final_loop_optimizations.apply(transformed_fn)
    _transform_cache[key] = transformed_fn
    if transformed_fn is not fn:
      key = transformed_fn.cache_key
      _transform_cache[key] = transformed_fn
  if value_specialization: 
    transformed_fn = specialize(transformed_fn, args)
    key = transformed_fn.cache_key 
  
   
  compiled_fn = PyModuleCompiler().compile_entry(fn)
  assert len(args) == len(fn.input_types)
  result = compiled_fn.c_fn(*args)
  return result