from .. import names, prims  
from ..ndtypes import ScalarT, Type, type_conv  
from .. syntax import  FormalArgs, Var, UntypedFn, Return, PrimCall, Expr, Cast 


_untyped_fn_cache = {}
def simple_untyped_fn(name, 
                        expr,
                        n_inputs = 1, 
                        fixed_args  = [], 
                        keyword_args = {},                         
                        unpack = False):
  key = name, expr, n_inputs, tuple(fixed_args), tuple(keyword_args.items()), unpack
  if key in _untyped_fn_cache:
    return _untyped_fn_cache[key]
  
  fn_name = names.fresh(name)
  args_obj = FormalArgs()

  arg_vars = []
  for name in names.fresh_list(n_inputs):
    args_obj.add_positional(name)
    arg_vars.append(Var(name))
  
  if unpack:
    combined_args = tuple(fixed_args) + tuple(arg_vars)
  else:
    combined_args = tuple(fixed_args) + (tuple(arg_vars),)
  result = expr(*combined_args, **keyword_args)
  body = [Return(result)]
  fundef = UntypedFn(fn_name, args_obj, body, [])
  _untyped_fn_cache[key] = fundef 
  return fundef 
  
def build_untyped_prim_fn(p):
  """Given a primitive, return an untyped function which calls that prim"""
  assert isinstance(p, prims.Prim), "Expected Prim but got %s" % p 
  return simple_untyped_fn(p.name, PrimCall, p.nin, [p])


def build_untyped_expr_fn(expr, n_args = 1):
  """Given an expression, return a function which applies that expression to arguments"""
  return simple_untyped_fn(expr.__name__ + "_fn", expr, n_args)
    
_untyped_cast_wrappers = {}
def build_untyped_cast_fn(t):
  if not isinstance(t, Type):
    t = type_conv.equiv_type(t)
  assert isinstance(t, ScalarT), "Expected scalar type but got %s" % t 
  return simple_untyped_fn("cast_" + str(t), Cast, 1, keyword_args = {'type': t})
  