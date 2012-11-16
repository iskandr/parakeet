import syntax_helpers
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
