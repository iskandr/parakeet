from prepare_args import prepare_args
from ..transforms.pipeline  import lower_to_loops
from ..value_specialization import specialize
from ..config import value_specialization
from pymodule_compiler import PyModuleCompiler 

def run(fn, args):  
  args = prepare_args(args, fn.input_types)

  fn = lower_to_loops(fn)
  
  if value_specialization: 
    fn = specialize(fn, args)
  
  compiled_fn = PyModuleCompiler().compile_entry(fn)
  c_fn = compiled_fn.c_fn 
  return c_fn(*args)

"""
_cache = {}
def run(fn, args):
  args = prepare_args(args, fn.input_types)

  fn = lower_to_loops(fn)
  
  if value_specialization: 
    fn = specialize(fn, args)
  
  key = fn.cache_key
  if key in _cache:
    c_fn = _cache[key]
  else:
    compiled_fn = PyModuleCompiler().compile_entry(fn)
    c_fn = compiled_fn.c_fn 
    _cache[key] = c_fn 
  return c_fn(*args)
"""