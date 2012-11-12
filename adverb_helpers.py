import syntax
import syntax_helpers
import names
import args
import adverbs
import ast_conversion
import function_registry

_adverb_wrapper_cache = {}
def untyped_wrapper(adverb_class, arg_names = ['fn', 'x'],  axis = 0):
  # print "untyped_wrapper", adverb_class, arg_names, axis
  axis = syntax_helpers.wrap_if_constant(axis)
  key = adverb_class, tuple(arg_names), axis
  if key in _adverb_wrapper_cache:
    return _adverb_wrapper_cache[key]
  else:
    local_arg_names = map(names.refresh, arg_names)

    local_arg_vars = map(syntax.Var, local_arg_names)
    fn_var = local_arg_vars[0]
    data_vars = local_arg_vars[1:]
    adverb = adverb_class(fn = fn_var, args = data_vars, axis = axis)
    body = [syntax.Return(adverb)]
    fn_name = names.fresh(adverb_class.node_type() + "_wrapper")
    fundef = syntax.Fn(fn_name, args.Args(positional = local_arg_names), body)
    function_registry.untyped_functions[fn_name] = fundef
    _adverb_wrapper_cache[key] = fundef
    return fundef

_adverb_registry = {}
def is_registered_adverb(fn):
  return fn in _adverb_registry

def register_adverb(python_fn, wrapper):
  _adverb_registry[python_fn] = wrapper

def get_adverb_wrapper(python_fn):
  return _adverb_registry[python_fn]

def untyped_map_wrapper(fundef, axis = None):
  if not isinstance(fundef, syntax.Fn):
    fundef = ast_conversion.translate_function_value(fundef)
  assert len(fundef.args.defaults) == 0
  arg_names = ['fn'] + list(fundef.args.positional)
  return untyped_wrapper(adverbs.Map, arg_names, axis = axis)

def untyped_allpairs_wrapper(fundef, axis = None):
  if not isinstance(fundef, syntax.Fn):
    fundef = ast_conversion.translate_function_value(fundef)
  assert len(fundef.args.defaults) == 0
  assert len(fundef.args.positional) == 2
  arg_names = ['fn'] + list(fundef.args.positional)
  return untyped_wrapper(adverbs.AllPairs, arg_names, axis = axis)

def untyped_reduce_wrapper(fundef, axis = None):
  if not isinstance(fundef, syntax.Fn):
    fundef = ast_conversion.translate_function_value(fundef)
  assert len(fundef.args.defaults) == 0
  arg_names = ['fn'] + list(fundef.args.positional)
  return untyped_wrapper(adverbs.Reduce, arg_names, axis = axis)

def untyped_scan_wrapper(fundef, axis = None):
  if not isinstance(fundef, syntax.Fn):
    fundef = ast_conversion.translate_function_value(fundef)
  assert len(fundef.args.defaults) == 0
  arg_names = ['fn'] + list(fundef.args.positional)
  return untyped_wrapper(adverbs.Scan, arg_names, axis = axis)


import core_types
import array_type
def max_rank(arg_types):
  """
  Given a list of types, find the maximum rank of the list
  and also check that all other types have either the same rank
  or are scalars
  """
  curr_max = 0
  for t in arg_types:
    if isinstance(t, array_type.ArrayT):
      assert curr_max == 0 or curr_max == t.rank,  \
       "Adverb can't accept inputs of rank %d and %d" % (curr_max, t.rank)
      curr_max = t.rank
    return curr_max

def max_rank_arg(args):
  """
  Given a list of arguments, return one which has the maximum rank
  """
  r = max_rank(syntax_helpers.get_types(args))
  for arg in args:
    if arg.type.rank == r:
      return arg



def num_outer_axes(arg_types, axis):
  """
  Helper for adverb type inference to figure out
  how many axes it will loop over -- either 1 particular
  one or all of them when axis is None. 
  """
  axis = syntax_helpers.unwrap_constant(axis)
  if isinstance(arg_types, core_types.Type):
    max_arg_rank = arg_types.rank 
  else:
    max_arg_rank = max_rank(arg_types)
  return 1 if (max_arg_rank > 0 and axis is not None) else max_arg_rank
