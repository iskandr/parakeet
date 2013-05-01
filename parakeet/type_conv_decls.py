import numpy as np
import types 

import core_types 
import prims  
import type_conv

from array_type import make_array_type, ArrayT
from closure_type import make_closure_type, ClosureT
from core_types import NoneT, NoneType, TypeValueT, from_dtype
from decorators import jit, macro 
from tuple_type import make_tuple_type, TupleT 

type_conv.register(type(None), NoneT, lambda _: NoneType)
type_conv.register([np.dtype], TypeValueT, lambda dt: TypeValueT(from_dtype(dt))) 

def typeof_type(t):
  assert hasattr(t, 'dtype'), "Can only convert numpy types"
  dt = t(0).dtype 
  pt = from_dtype(dt)
  return TypeValueT(pt) 
type_conv.register(types.TypeType, TypeValueT, typeof_type)

def typeof_tuple(python_tuple):
  return make_tuple_type(map(type_conv.typeof, python_tuple))

type_conv.register(types.TupleType, TupleT, typeof_tuple)

def typeof_array(x):
  x = np.asarray(x)
  elt_t = core_types.from_dtype(x.dtype)
  rank = len(x.shape)
  return make_array_type(elt_t, rank)

type_conv.register((np.ndarray, list, xrange), ArrayT, typeof_array)

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
