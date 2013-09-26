from prepare_args import prepare_args
from ..transforms.pipeline  import loopify, flatten 
from ..transforms.stride_specialization import specialize
from ..config import stride_specialization
from compiler import compile_entry 

def run(fn, args):
  args = prepare_args(args, fn.input_types)
  fn = loopify.apply(fn)
  fn = flatten(fn)
  if stride_specialization:
    fn = specialize(fn, python_values = args)
  compiled_fn = compile_entry(fn)
  assert len(args) == len(fn.input_types)
  result = compiled_fn.c_fn(*args)
  return result