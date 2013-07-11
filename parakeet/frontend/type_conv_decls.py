import types 

from loopjit.ndtypes import ArrayT,  make_closure_type, ClosureT
from loopjit.ndtypes import type_conv, typeof_array, 
from loopjit import prims 

from frontend import jit, macro 

# ndtypes already register NumPy arrays type converters, 
# but parakeet also treats ranges and lists as arrays 
type_conv.register((list, xrange), ArrayT, typeof_array)

def typeof_prim(p):
  untyped_fn = prims.prim_wrapper(p)
  return make_closure_type(untyped_fn, [])

type_conv.register(prims.class_list, ClosureT, typeof_prim)

def typeof_fn(f):
  import ast_conversion
  untyped_fn = ast_conversion.translate_function_value(f)
  closure_args = untyped_fn.python_nonlocals()
  closure_arg_types = map(type_conv.typeof, closure_args)
  return make_closure_type(untyped_fn, closure_arg_types)

type_conv.register(types.FunctionType, ClosureT, typeof_fn)
type_conv.register(jit, ClosureT, typeof_fn)
type_conv.register(macro, ClosureT, typeof_fn)
type_conv.register(types.BuiltinFunctionType, ClosureT, typeof_fn)
