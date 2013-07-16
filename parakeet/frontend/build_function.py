from treelike import NestedBlocks

from .. import names, syntax
from ..ndtypes import NoneType 
from ..syntax import TypedFn, Var  
from ..transforms.builder import Builder 


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
  name = names.fresh(name)
  
  input_names_with_types = zip(input_names, input_types)
  
  f = TypedFn(name = name, 
              arg_names = input_names, 
              body = [], 
              type_env = dict(input_names_with_types),
              input_types = input_types, 
              return_type = return_type) 
              
  input_vars = [Var(arg_name, t) for arg_name, t in input_names_with_types]
  blocks = NestedBlocks()
  blocks.push(f.body)
  builder = Builder(type_env = f.type_env, blocks = blocks)
  return f, input_vars, builder 