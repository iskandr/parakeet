
from .. import config, type_inference
from .. ndtypes import type_conv
from ..syntax import UntypedFn, TypedFn
from ..syntax.fn_args import ActualArgs
from .. transforms import pipeline
 
def prepare_args(fn, args, kwargs):
  """
  Fetch the function's nonlocals and return an ActualArgs object of both the arg
  values and their types
  """
  assert isinstance(fn, UntypedFn)
   

  nonlocals = list(fn.python_nonlocals())
  arg_values = ActualArgs(nonlocals + list(args), kwargs)

  # get types of all inputs
  arg_types = arg_values.transform(type_conv.typeof)
  return arg_values, arg_types


def specialize(untyped, args, kwargs = {}):
  """
  Translate, specialize and begin to optimize the given function for the types
  of the supplies arguments.

  Return the untyped and typed representations, along with all the
  arguments in a linear order. 
  """

  arg_values, arg_types = prepare_args(untyped, args, kwargs)
  
  # convert the awkward mix of positional, named, and starargs 
  # into a positional sequence of arguments
  linear_args = untyped.args.linearize_without_defaults(arg_values)
  
  # propagate types through function representation and all
  # other functions it calls
  typed_fn = type_inference.specialize(untyped, arg_types)
  
  from .. transforms.pipeline import high_level_optimizations
  # apply high level optimizations 
  optimized_fn = high_level_optimizations.apply(typed_fn)
  return optimized_fn, linear_args 
  
def run_typed_fn(fn, args, backend = None):
  assert isinstance(fn, TypedFn)
  actual_types = tuple(type_conv.typeof(arg) for arg in  args)
  expected_types = fn.input_types
  assert actual_types == expected_types, \
    "Arg type mismatch, expected %s but got %s" % \
    (expected_types, actual_types)

  if backend is None:
    backend = config.default_backend
    
  if backend == 'llvm':
    from ..llvm_backend import ctypes_to_generic_value, generic_value_to_python, compile_fn 
    from ..llvm_backend.llvm_context import global_context
    exec_engine = global_context.exec_engine
    lowered_fn = pipeline.lowering.apply(fn)
    llvm_fn = compile_fn(lowered_fn).llvm_fn

    # calling conventions are that output must be preallocated by the caller'
    ctypes_inputs = [t.from_python(v) for (v,t) in zip(args, expected_types)]
    gv_inputs = [ctypes_to_generic_value(cv, t) for (cv,t) in
               zip(ctypes_inputs, expected_types)]

    gv_return = exec_engine.run_function(llvm_fn, gv_inputs)
    return generic_value_to_python(gv_return, fn.return_type)
  
  elif backend == "interp":
    from .. import interp 
    return interp.eval_fn(fn, args)
  
  elif backend == "shiver":
    
    from ..llvm_backend.llvm_context import global_context
    exec_engine = global_context.exec_engine
    # import shiver 
    # shiver.parfor(fn, niters, fixed_args, ee = exec_engine)
    assert False, "Shiver not yet implemented"
  else:
    assert False, "Unknown backend %s" % backend 

def run_untyped_fn(fn, args, kwargs = None, backend = None):
  assert isinstance(fn, UntypedFn)
  if kwargs is None:
    kwargs = {}
  typed_fn, linear_args = specialize(fn, args, kwargs)
  return run_typed_fn(typed_fn, linear_args, backend)
  
def run_python_fn(fn, args, kwargs = None, backend = None):
  """
  Given a python function, run it in Parakeet on the supplied args
  """
  if kwargs is None:
    kwargs = {}
  import ast_conversion
  # translate from the Python AST to Parakeet's untyped format
  untyped = ast_conversion.translate_function_value(fn)
  return run_untyped_fn(untyped, args, kwargs, backend)
  