
   
from .. import names 
from .. ndtypes import ArrayT, Type

from adverbs import Map
from expr import Var
from formal_args import FormalArgs
from helpers import  unwrap_constant, zero_i64, get_types
from stmt import Return 
from untyped_fn import UntypedFn

def max_rank(arg_types):
  """
  Given a list of types, find the maximum rank of the list and also check that
  all other types have either the same rank or are scalars
  """

  curr_max = 0
  for t in arg_types:
    if isinstance(t, ArrayT):
      r =  t.rank
      if r > curr_max:
        curr_max = r  
      #assert curr_max == 0 or curr_max == t.rank,  \
      # "Adverb can't accept inputs of rank %d and %d" % (curr_max, t.rank)
      
  return curr_max

def max_rank_arg(args):
  """Given a list of arguments, return one which has the maximum rank"""

  r = max_rank(get_types(args))
  for arg in args:
    if arg.type.rank == r:
      return arg

def num_outer_axes(arg_types, axis):
  """
  Helper for adverb type inference to figure out how many axes it will loop over
  -- either 1 particular one or all of them when axis is None.
  """

  axis = unwrap_constant(axis)
  if isinstance(arg_types, Type):
    max_arg_rank = arg_types.rank
  else:
    max_arg_rank = max_rank(arg_types)
  return 1 if (max_arg_rank > 0 and axis is not None) else max_arg_rank

_nested_map_cache = {}
def nested_maps(inner_fn, depth, arg_names):
  
  if depth <= 0:
    assert isinstance(inner_fn, UntypedFn)
    return inner_fn

  key = inner_fn.name, depth, tuple(arg_names)
  if key in _nested_map_cache:
    return _nested_map_cache[key]
  args_obj = FormalArgs()
  arg_vars = []
  for var_name in arg_names:
    local_name = names.refresh(var_name)
    args_obj.add_positional(local_name)
    arg_vars.append(Var(local_name))

  name = names.fresh(inner_fn.name + "_broadcast%d" % depth)
  nested_fn = nested_maps(inner_fn, depth - 1, arg_names)

  map_expr = Map(fn = nested_fn, 
                 axis = zero_i64, 
                 args = arg_vars)
  fn = UntypedFn(
    name = name,
    args = args_obj,
    body = [Return(map_expr)]
  )

  _nested_map_cache[key] = fn
  return fn



