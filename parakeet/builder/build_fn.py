from treelike import NestedBlocks

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