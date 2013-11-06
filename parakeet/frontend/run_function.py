
from .. import config, type_inference 
from ..analysis import contains_loops 
from ..ndtypes import type_conv, Type 
from ..syntax import UntypedFn, TypedFn, ActualArgs
from ..transforms import pipeline

def prepare_args(fn, args, kwargs):
  """
  Fetch the function's nonlocals and return an ActualArgs object of both the arg
  values and their types
  """
  assert not isinstance(fn, TypedFn), "[prepare_args] Only works for untyped functions"
  if not isinstance(fn, UntypedFn):
    import ast_conversion
    fn = ast_conversion.translate_function_value(fn)   
    
  nonlocals = list(fn.python_nonlocals())
  arg_values = ActualArgs(nonlocals + list(args), kwargs)

  # get types of all inputs
  def _typeof(arg):
    if hasattr(arg, 'type') and isinstance(arg.type, Type):
      return arg.type 
    else:
      return type_conv.typeof(arg)
  arg_types = arg_values.transform(_typeof)
  return arg_values, arg_types
  

def specialize(untyped, args, kwargs = {}, optimize = True):
  """
  Translate, specialize and begin to optimize the given function for the types
  of the supplies arguments.

  Return the untyped and typed representations, along with all the
  arguments in a linear order. 
  """

  if not isinstance(untyped, UntypedFn):
    import ast_conversion
    untyped = ast_conversion.translate_function_value(untyped)
       
  arg_values, arg_types = prepare_args(untyped, args, kwargs)
  
  # convert the awkward mix of positional, named, and starargs 
  # into a positional sequence of arguments
  linear_args = untyped.args.linearize_without_defaults(arg_values)
  
  # propagate types through function representation and all
  # other functions it calls
   
  typed_fn = type_inference.specialize(untyped, arg_types)
  if optimize: 
    from .. transforms.pipeline import normalize 
    # apply high level optimizations 
    typed_fn = normalize.apply(typed_fn)
  return typed_fn, linear_args 

def run_typed_fn(fn, args, backend = None):
  
  assert isinstance(fn, TypedFn)
  actual_types = tuple(type_conv.typeof(arg) for arg in  args)
  expected_types = fn.input_types
  assert actual_types == expected_types, \
    "Arg type mismatch, expected %s but got %s" % \
    (expected_types, actual_types)

  
  if backend is None:
    backend = config.backend
    
  if backend == 'c':
    from .. import c_backend
    return c_backend.run(fn, args)
   
  elif backend == 'openmp':
    from .. import openmp_backend 
    return openmp_backend.run(fn, args)
  
  elif backend == 'cuda':
    from .. import cuda_backend 
    return cuda_backend.run(fn, args)
  
  elif backend == 'llvm':
    from ..llvm_backend.llvm_context import global_context
    from ..llvm_backend import generic_value_to_python 
    from ..llvm_backend import ctypes_to_generic_value, compile_fn 
    lowered_fn = pipeline.lowering.apply(fn)
    llvm_fn = compile_fn(lowered_fn).llvm_fn

    ctypes_inputs = [t.from_python(v) 
                   for (v,t) 
                   in zip(args, expected_types)]
    gv_inputs = [ctypes_to_generic_value(cv, t) 
                 for (cv,t) 
                 in zip(ctypes_inputs, expected_types)]
    exec_engine = global_context.exec_engine
    gv_return = exec_engine.run_function(llvm_fn, gv_inputs)
    return generic_value_to_python(gv_return, fn.return_type)

  elif backend == "interp":
    from .. import interp 
    fn = pipeline.loopify(fn)
    return interp.eval_fn(fn, args)
  
  else:
    assert False, "Unknown backend %s" % backend 

def run_untyped_fn(fn, args, kwargs = None, backend = None):
  assert isinstance(fn, UntypedFn)
  if kwargs is None:
    kwargs = {}
  typed_fn, linear_args = specialize(fn, args, kwargs)
  return run_typed_fn(typed_fn, linear_args, backend)

def run_python_ast(fn_name, fn_args, fn_body, globals_dict, 
                     arg_values, kwarg_values = None, backend = None):
  """
  Instead of giving Parakeet a function to parse, you can construct a Python 
  AST yourself and then have that converted into a Typed Parakeet function
  """
  import ast_conversion
  untyped = ast_conversion.translate_function_ast(fn_name, fn_args, fn_body, globals_dict)
  return run_untyped_fn(untyped, arg_values, kwarg_values, backend)
    
def run_python_fn(fn, args, kwargs = None, backend = None):
  """
  Given a python function, run it in Parakeet on the supplied args
  """
  import ast_conversion
  # translate from the Python AST to Parakeet's untyped format
  untyped = ast_conversion.translate_function_value(fn)
  return run_untyped_fn(untyped, args, kwargs, backend)
  
