
from .. import names 
from ..ndtypes import ClosureT, make_closure_type
from ..syntax import Expr, UntypedFn, Closure, Var, Return, ClosureElt, TypedFn, FormalArgs  
from ..syntax.helpers import get_types 

def unpack_closure(closure):
  """
  Given an object which could be either a function, a function's name, a
  closure, or a closure type:
  Return the underlying untyped function and the closure arguments
  """

  if closure.__class__ is ClosureT:
    fn, closure_args = closure.fn, closure.arg_types
  elif closure.__class__ is Closure:
    fn = closure.fn 
    closure_args = closure.args 
  elif closure.type.__class__ is ClosureT:
    fn, arg_types = closure.type.fn, closure.type.arg_types
    closure_args = \
        [ClosureElt(closure, i, type = arg_t)
         for (i, arg_t) in enumerate(arg_types)]
  else:
    fn = closure
    closure_args = []
    # fn = UntypedFn.registry[fn]
  return fn, closure_args


def make_typed_closure(clos, typed_fn):
  if clos.__class__ is UntypedFn:
    return typed_fn

  assert isinstance(clos, Expr) and clos.type.__class__ is ClosureT
  _, closure_args = unpack_closure(clos)
  if len(closure_args) == 0:
    return typed_fn
  else:
    t = make_closure_type(typed_fn, get_types(closure_args))
    return Closure(typed_fn, closure_args, t)

def mk_untyped_identity():
  var_name = names.fresh('x')
  var_expr = Var(var_name)
  fn_name = names.fresh('identity')
  args_obj = FormalArgs()
  args_obj.add_positional(var_name)
  return UntypedFn(name = fn_name, 
                   args = args_obj, 
                   body = [Return(var_expr)])

untyped_identity_function = mk_untyped_identity()

def _get_fundef(fn):
  c = fn.__class__ 
  if c is UntypedFn or c is TypedFn:
    return fn
  assert c is str, "Unexpected function %s : %s"  % (fn, fn.type)
  return UntypedFn.registry[fn]

def _get_closure_type(fn):
  assert isinstance(fn, (UntypedFn, TypedFn, ClosureT, Closure, Var)), \
    "Expected function, got %s" % fn
  c = fn.__class__ 
  if c is ClosureT:
    return fn
  elif c is Closure:
    return fn.type
  elif c is Var:
    assert fn.type.__class__ is ClosureT
    return fn.type
  else:
    fundef = _get_fundef(fn)
    return make_closure_type(fundef, [])