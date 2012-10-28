
import syntax 
import syntax_helpers
import names 

import adverbs 

_adverb_wrapper_cache = {}
def untyped_wrapper(adverb_class, untyped_fn, **adverb_params):
  """
  Given an adverb and its untyped function argument, 
  create a fresh function whose body consists of only 
  the evaluation of the adverb.
  
  The wrapper should accept the same number of arguments 
  as the parameterizing function and correctly forward any
  global dependencies by creating a closure.  
  """
  
  fn_args = untyped_fn.args.fresh_copy()
  
  nonlocals = untyped_fn.nonlocals
  
  # create a local closure to forward nonlocals into the adverb 
  n_nonlocals = len(nonlocals)
  # I'm still indecisive whether args to functions should be strings & tuples
  # or syntax nodes so, to be defensive, just accomodate both cases here 
  closure_args = syntax_helpers.wrap_vars(fn_args.positional[:n_nonlocals])
  closure = syntax.Closure(untyped_fn.name, closure_args)
  closure_var = syntax.Var(names.fresh("closure"))
  body = [syntax.Assign(closure_var, closure)] 
  
  # the adverb parameters are given as python values, convert them to
  # constant syntax nodes 
  adverb_param_exprs = {}
  for (k,v) in adverb_params.items():
    adverb_param_exprs[k] = syntax_helpers.wrap_if_constant(v)
  adverb_args = syntax_helpers.wrap_vars(fn_args.positional[n_nonlocals:])
  adverb = adverb_class(closure_var, adverb_args, **adverb_param_exprs)
  body += [syntax.Return(adverb)]
  fn_name = names.fresh(adverb_class.node_type() + "_wrapper")
  return syntax.Fn(fn_name, fn_args, body, nonlocals)

def untyped_map_wrapper(untyped_fn, axis = None):
  return untyped_wrapper(adverbs.Map, untyped_fn, axis = axis)

def untyped_allpairs_wrapper(untyped_fn, axis = None):
  return untyped_wrapper(adverbs.AllPairs, untyped_fn, axis = axis)

def untyped_reduce_wrapper(untyped_fn,  axis = None, init = None):
  # TODO: Add options for a combiner! 
  return untyped_wrapper(adverbs.Reduce, untyped_fn, axis = axis, init = init)

def untyped_scan_wrapper(untyped_fn,  axis = None, init = None):
  # TODO: Add options for a combiner! 
  return untyped_wrapper(adverbs.Scan, untyped_fn, axis = axis, init = init)


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
  
    
def lower_arg_rank(t, r):
  if isinstance(t, core_types.ScalarT):
    return t
  else:
    assert isinstance(t, array_type.ArrayT)
    assert t.rank >= r
    return array_type.make_array_type(t.elt_type, t.rank - r)

def increase_rank(t, r):
  if isinstance(t, core_types.ScalarT):
    return array_type.make_array_type(t, r)
  else:
    assert isinstance(t, array_type.ArrayT)
    return array_type.make_array_type(t.elt_type, t.rank + r)
        
def lower_arg_ranks(arg_types, r):
  return [lower_arg_rank(t, r) for t in arg_types]
    