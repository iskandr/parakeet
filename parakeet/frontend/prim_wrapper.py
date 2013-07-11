import names
import syntax

from args import FormalArgs


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
      arg_vars.append(syntax.Var(name))
    body = [syntax.Return(syntax.PrimCall(p, arg_vars))]
    fundef = syntax.Fn(fn_name, args_obj, body, [])
    _untyped_prim_wrappers[p] = fundef

    return fundef