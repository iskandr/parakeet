from prepare_args import prepare_args
from ..transforms import pipeline
from compiler import compile_entry 

def run(fn, args):
  args = prepare_args(args, fn.input_types)
  fn = pipeline.loopify.apply(fn)
  fn = pipeline.flatten(fn)
  #from ..transforms import  stride_specialization
  #flat_fn = stride_specialization.specialize(flat_fn, args)
  compiled_fn = compile_entry(fn)
  assert len(args) == len(fn.input_types)
  result = compiled_fn.c_fn(*args)
  return result