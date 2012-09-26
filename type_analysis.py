import syntax
import ptype

def match(pattern, t, env):
  """
  Given a left-hand-side of tuples & vars, 
  a right-hand-side of tuples & types, 
  traverse the tuple structure recursively and
  put the matched variable names in an environment
  """
  if isinstance(pattern, syntax.Var):
    env[pattern.name] = t
  elif isinstance(pattern, syntax.Tuple):
    assert isinstance(t, ptype.Tuple)
    pat_elts = pattern.elts
    type_elts = t.elts 
    assert len(pat_elts) == len(type_elts), \
      "Mismatch between expected and given number of values"
    for (pi , ti) in zip(pat_elts, type_elts):
      match(pi, ti, env)
  else:
    raise RuntimeError("Unexpected pattern: %s" % pattern)    

class InferenceFailed(Exception):
  def __init__(self, msg):
    self.msg = msg 

import copy 
def typed(untyped_obj, t):
  typed_obj = copy.deepcopy(untyped_obj)
  setattr(typed_obj, 'type', t)
  return typed_obj 

def infer_return_types(untyped_id, arg_types):
        

def specialize_fn(fn, arg_types):
  tenv = {}
  match(fn.args, arg_types, tenv)
  
  def expr_type(expr):
    def expr_Closure():
      arg_types = map(expr_type, expr.args)
      closure_type = ptype.Closure(expr.fn, arg_types)
      closure_set = ptype.ClosureSet(closure_type)
      return closure_set 
    
    def expr_Invoke():
      closure_set = expr_type(expr.closure)
      arg_types = map(expr_type, expr.args)
      
      for closure_type in closure_set.closures:
        untyped_id, closure_arg_types = closure_type.fn, closure_type.args
        ret = infer_return_types(untyped_id, closure_arg_types + arg_types)
        if result_type is None:
          result_type = ret
        elif isinstance(result_type, ptype.ClosureSet) and \
             isinstance(ret, ptype.ClosureSet):
          result_type = ptype.ClosureSet(result_type.closures.union(ret.closures))
        elif result_type != ret: 
          raise InferenceFailed("Call might result in either %s or %s" % (result_type, ret) )
      return result_type
    
    nodetype = expr.__class__.__name__   
    fn_name = 'expr_' + nodetype   
    local_fns = locals()
    if fn_name not in local_fns:
      raise RuntimeError("Unsupported node type %s" % nodetype)
    else:
      result_type = local_fns[fn_name]
      expr.type = result_type 
        