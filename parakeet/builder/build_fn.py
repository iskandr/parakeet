from dsltools import NestedBlocks

from .. import names, syntax
from ..ndtypes import NoneType 
from ..syntax import TypedFn
from ..builder import Builder 


def fresh_builder(fn):
  blocks = NestedBlocks()
  blocks.push(fn.body)
  return Builder(type_env = fn.type_env, blocks = blocks)
  
def build_fn(input_types, 
             return_type = NoneType, 
             name = None, 
             input_names = None):
  n_inputs = len(input_types)
  if input_names is None:
    input_names = syntax.gen_data_arg_names(n_inputs)
  assert len(input_names) == n_inputs 
  if name is None:
    name = 'f'
  name = names.refresh(name)
 
  f = TypedFn(name = name, 
              arg_names = input_names, 
              body = [], 
              type_env = dict(zip(input_names, input_types)),
              input_types = input_types, 
              return_type = return_type) 
              
  builder = fresh_builder(f)
  input_vars = builder.input_vars(f) 
  return f, builder, input_vars  


_identity_fn_cache = {}
def mk_identity_fn(t):
  if t in _identity_fn_cache:
    return _identity_fn_cache[t]
  f, b, (x,) = build_fn([t], t, name = "ident")
  b.return_(x)
  _identity_fn_cache[t] = f
  return f

_cast_fn_cache = {}
def mk_cast_fn(from_type, to_type):
  key = (from_type, to_type)
  if key in _cast_fn_cache:
    return _cast_fn_cache[key]
  f, b, (x,) = build_fn([from_type], to_type, name = "cast_%s_%s" % (from_type, to_type))
  b.return_(b.cast(x, to_type))
  _cast_fn_cache[key] = f
  return f

_prim_fn_cache = {}
def mk_prim_fn(prim, arg_types):
  key = prim, tuple(arg_types)
  if key in _prim_fn_cache:
    return _prim_fn_cache[key]
  
  upcast_types = prim.expected_input_types(arg_types)
  result_type = prim.result_type(upcast_types)
  f, b, args = build_fn(arg_types, result_type, name =  prim.name)
  # builder will auto-cast argument vars to appropriate types for the given primitive
  b.return_(b.prim(prim, args))
  _prim_fn_cache[key] = f 
  return f 
  
