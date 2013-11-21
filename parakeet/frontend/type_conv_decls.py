import numpy as np
from numpy import  asarray 
import types 


from ..syntax import UntypedFn, TypedFn 
from ..ndtypes import (scalar_types, 
                       make_array_type, ArrayT, 
                       make_tuple_type, TupleT, 
                       make_closure_type, ClosureT, 
                       NoneT, NoneType, TypeValueT)
from ..ndtypes.type_conv import typeof, register   

register(type(None), NoneT, lambda _: NoneType)


def typeof_dtype(dt):
  return TypeValueT(scalar_types.from_dtype(dt))

register([np.dtype], TypeValueT, typeof_dtype) 

def typeof_type(t):
  assert hasattr(t, 'dtype'), "Can only convert numpy types"
  dt = t(0).dtype 
  pt = scalar_types.from_dtype(dt)
  return TypeValueT(pt) 

register(types.TypeType, TypeValueT, typeof_type)

def typeof_tuple(python_tuple):
  return make_tuple_type([typeof(elt_t) for elt_t in python_tuple])

register(types.TupleType, TupleT, typeof_tuple)

def typeof_array(x):
  x = asarray(x)
  elt_t = scalar_types.from_dtype(x.dtype)
  return make_array_type(elt_t, x.ndim)

register(np.ndarray, ArrayT, typeof_array)

from .. import prims 
register((list, xrange), ArrayT, typeof_array)



from decorators  import jit, macro 

 
# ndtypes already register NumPy arrays type converters, 
# but parakeet also treats ranges and lists as arrays 
register((list, xrange), ArrayT, typeof_array)

def typeof_prim(p):
  from ..syntax.wrappers import  build_untyped_prim_fn
  untyped_fn = build_untyped_prim_fn(p)
  return make_closure_type(untyped_fn, ())

register(prims.class_list, ClosureT, typeof_prim)

def typeof_fn(f):
  import ast_conversion
  untyped_fn = ast_conversion.translate_function_value(f)
  closure_args = untyped_fn.python_nonlocals()
  closure_arg_types = map(typeof, closure_args)
  return make_closure_type(untyped_fn, closure_arg_types)


register(types.FunctionType, ClosureT, typeof_fn)
register(jit, ClosureT, typeof_fn)
register(macro, ClosureT, typeof_fn)
register(types.BuiltinFunctionType, ClosureT, typeof_fn)
register(np.ufunc, ClosureT, typeof_fn)
register(UntypedFn, ClosureT, typeof_fn)
register(TypedFn, ClosureT, typeof_fn)
