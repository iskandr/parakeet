from .. import names 
from .. syntax import  FormalArgs, Var, UntypedFn, Return, PrimCall  


_untyped_prim_wrappers = {}
def prim_wrapper(p):
  """Given a primitive, return an untyped function which calls that prim"""
  if p in _untyped_prim_wrappers:
    return _untyped_prim_wrappers[p]
  else:
    fn_name = names.fresh(p.name)

    args_obj = FormalArgs()

    arg_vars = []
    for name in names.fresh_list(p.nin):
      args_obj.add_positional(name)
      arg_vars.append(Var(name))
    body = [Return(PrimCall(p, arg_vars))]
    fundef = UntypedFn(fn_name, args_obj, body, [])
    _untyped_prim_wrappers[p] = fundef

    return fundef