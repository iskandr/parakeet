from args import FormalArgs, ActualArgs

import config 
import names
import syntax
import syntax_helpers
import type_conv 
import type_inference 

def prepare_args(fn, args, kwargs):
  """
  Fetch the function's nonlocals and return an ActualArgs object of both the arg
  values and their types
  """
  if isinstance(fn, syntax.Fn):
    untyped = fn
  else:
    import ast_conversion
    # translate from the Python AST to Parakeet's untyped format
    untyped = ast_conversion.translate_function_value(fn)

  nonlocals = list(untyped.python_nonlocals())
  arg_values = ActualArgs(nonlocals + list(args), kwargs)

  # get types of all inputs
  arg_types = arg_values.transform(type_conv.typeof)
  return untyped, arg_values, arg_types

def specialize(fn, args, kwargs = {}):
  """
  Translate, specialize and begin to optimize the given function for the types
  of the supplies arguments.

  Return the untyped and typed representations, along with all the
  arguments in a linear order. 
  """

  untyped, arg_values, arg_types = prepare_args(fn, args, kwargs)
  
  # convert the awkward mix of positional, named, and starargs 
  # into a positional sequence of arguments
  linear_args = untyped.args.linearize_without_defaults(arg_values)
  
  # propagate types through function representation and all
  # other functions it calls
  typed_fn = type_inference.specialize(untyped, arg_types)
  
  from .. transforms.pipeline import high_level_optimizations
  # apply high level optimizations 
  optimized_fn = high_level_optimizations.apply(typed_fn)
  
  return untyped, optimized_fn, linear_args 
 
  
def run(fn, *args, **kwargs):
  """
  Given a python function, run it in Parakeet on the supplied args
  """
  
  if '_mode' in kwargs:
    backend_name = kwargs['_mode']
    del kwargs['_mode']
  else:
    backend_name = config.default_backend
  
  _, typed_fn, linear_args = specialize(fn, args, kwargs)
 
  import backend
  assert hasattr(backend, backend_name), "Unrecognized backend %s" % backend_name
  b = getattr(backend, backend_name) 
  return b.run(typed_fn, linear_args)
  
