from itertools import izip 
import numpy as np
from ..ndtypes import (type_conv, ScalarT, ArrayT, FnT, ClosureT, SliceT, NoneT, TupleT, TypeValueT)
from ..syntax import TypedFn, UntypedFn


def prepare_closure_args(untyped_fn):
  closure_args = untyped_fn.python_nonlocals()
  closure_arg_types = [type_conv.typeof(v) for v in closure_args]
  return prepare_args(closure_args, closure_arg_types)
      

def prepare_arg(arg, t):
  if isinstance(t, ScalarT):
    return t.dtype.type(arg)
  elif isinstance(t, ArrayT):
    return np.asarray(arg)
  elif isinstance(t, TupleT):
    arg = tuple(arg)
    assert len(arg) == len(t.elt_types)
    return prepare_args(arg, t.elt_types)
  elif isinstance(t, (NoneT, SliceT)):
    return arg
  elif isinstance(t, TypeValueT):
    return ()
  elif isinstance(t, (FnT, ClosureT)):
    if isinstance(arg, TypedFn):
      return ()
    elif isinstance(arg, UntypedFn):
      return prepare_closure_args(arg)
    else:
      from ..frontend import ast_conversion
      untyped = ast_conversion.translate_function_value(arg)
      return prepare_closure_args(untyped)
  assert False, "Can't call compiled C code with argument %s (Python type = %s, Parakeet type = %s)" % \
    (arg, type(arg), t)
  
def prepare_args(args, arg_types):
  return tuple(prepare_arg(arg, t) for arg, t in izip(args, arg_types))
  

