

import types
import numpy as np

from ..syntax import TypedFn, UntypedFn

plain_data_types = (types.IntType, types.LongType, types.FloatType, types.BooleanType,
                    types.SliceType, types.TupleType, np.ndarray)

def is_plain_data(arg):
  return isinstance(arg, plain_data_types)

def prepare_arg(arg):
  if is_plain_data(arg):
    return arg
  elif isinstance(arg, list):
    return np.ndarray(arg)
  elif isinstance(arg, TypedFn):
    return ()
  elif isinstance(arg, UntypedFn):
    return prepare_args(arg.python_nonlocals())
  elif hasattr(arg, '__call__'):
    from ..frontend import ast_conversion
    untyped = ast_conversion.translate_function_value(arg)
    return prepare_args(untyped.python_nonlocals())
  else:
    assert False, "Can't call compiled C code with argument %s : %s" % (arg, type(arg))
    
def prepare_args(args):
  result = tuple(prepare_arg(arg) for arg in args)
  return result