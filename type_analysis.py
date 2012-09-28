import syntax
import ptype
import names 
from function_registry import untyped_functions, typed_functions
from common import dispatch 


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
    raise RuntimeError("Unexpected pattern %s %s : %s" % (pattern.__class__.__name__, pattern, t) )    

def match_args(arg_patterns, types):
  tenv = {}
  nargs = len(arg_patterns)
  ntypes = len(types)
  assert nargs == ntypes, \
    "Mismatch between %d args and %d input types" % (nargs, ntypes)
  for (p,t) in zip(arg_patterns, types):
    if isinstance(p, str):
      # TODO: get rid of this branch once all arg lists are uniformly either
      # Var or Tuple lists 

      match(syntax.Var(p), t, tenv)
    else:
      match(p, t, tenv)
  return tenv     

class InferenceFailed(Exception):
  def __init__(self, msg):
    self.msg = msg 


def infer_types(fn, arg_types):
  """
  Given a function definition and some input types, 
  return a type environment mapping variables to types
  and a return type. 
  This also implicitly specializes the function for the given
  types and caches this typed version. 
  """ 
  key = (fn.name, tuple(arg_types))
  if key in typed_functions:
    typed_fundef = typed_functions[key]
    return typed_fundef.return_type, typed_fundef.type_env
  else:
    type_env =  _infer_types(fn, arg_types) 
    return_type = type_env["$return"]
    typed_body = specialize(fn, type_env, return_type)
    typed_arg_names = map(names.refresh, fn.args)
    typed_id = names.refresh(fn.name)
    typed_fundef = syntax.TypedFn(name = typed_id, args = typed_arg_names, 
      body = typed_body, input_types = arg_types, return_type = return_type, 
      type_env = type_env)
    typed_functions[key] = typed_fundef 
    return return_type, type_env 
    

def specialize(fn, type_env, return_type):
  return None 



def _infer_types(fn, arg_types):
  """
  Actual implementation of type inference which doesn't attempt to 
  look up cached version of typed function
  """ 
  tenv = match_args(fn.args, arg_types)
  # keep track of the return 
  tenv['$return'] = ptype.Unknown
   
  def expr_type(expr):
    
    def expr_Closure():
      arg_types = map(expr_type, expr.args)
      closure_type = ptype.Closure(expr.fn, arg_types)
      closure_set = ptype.ClosureSet(closure_type)
      return closure_set 
    
    def expr_Invoke():
      closure_set = expr_type(expr.closure)
      arg_types = map(expr_type, expr.args)
      invoke_result_type = ptype.Unknown
      for closure_type in closure_set.closures:
        untyped_id, closure_arg_types = closure_type.fn, closure_type.args
        untyped_fundef = untyped_functions[untyped_id]
        ret, _ = infer_types(untyped_fundef, closure_arg_types + arg_types)
        invoke_result_type = invoke_result_type.combine(ret)
      return invoke_result_type
  
    def expr_PrimCall():
      arg_types = map(expr_type, expr.args)
     
      return ptype.combine_type_list(arg_types)
  
    def expr_Var():
      if expr.name in tenv:
        t = tenv[expr.name]
        return t
      else:
        raise names.NameNotFound(expr.name)
    
    def expr_Const():
      return ptype.type_of_value(expr.value)
    
    return dispatch(expr, prefix="expr")
  
  def merge_branches(phi_nodes):
    for result_var, (left_val, right_val) in phi_nodes.iteritems():
      left_type = expr_type(left_val)
      right_type = expr_type(right_val)
      tenv[result_var]  = left_type.combine(right_type)
  
  def analyze_stmt(stmt):
    def stmt_Assign():
      rhs_type = expr_type(stmt.rhs)
      match(stmt.lhs, rhs_type, tenv)
      
    def stmt_If():
      cond_type = expr_type(stmt.cond)
      assert isinstance(cond_type, ptype.Scalar), \
        "Condition has type %s but must be convertible to bool" % cond_type
      analyze_block(stmt.true)
      analyze_block(stmt.false)
      merge_branches(stmt.merge)
      
    def stmt_Return():
      t = expr_type(stmt.value)
      curr_return_type = tenv["$return"]
      tenv["$return"] = curr_return_type.combine(t)
    dispatch(stmt, prefix="stmt")
  
  def analyze_block(stmts):
    for stmt in stmts:
      analyze_stmt(stmt)
  analyze_block(fn.body)
  return tenv
      
      
      
      
#      
#      
#from parakeet_types import type_of_value 
#import syntax 
#from function_registry import typed_functions, untyped_functions
#
#class InferTypes(syntax.Traversal):
#  def stmt_Set(self, stmt, tenv): 
#    rhs_type = self.visit_expr(stmt.rhs, tenv)
#    tenv[stmt.lhs] = rhs_type
#
#  def expr_Const(self, expr, tenv):
#    return type_of_value(expr.value)
#
#  def expr_Var(self, expr, tenv):
#    assert expr.id in tenv, \
#      "Unknown variable " + expr.id  
#    return tenv[expr.id]
#  
#  def expr_Call(self, expr, tenv):
#    fn_expr = expr.fn
#    # list of types 
#    arg_types = map(self.visit_expr, expr.args)
#    # dict of string name -> type mappings
#    kwd_types = dict([ (k, self.visit_expr(v)) for \
#      (k,v) in expr.kwds.iteritems()])
#       
#    if isinstance(fn, syntax.Var):
#      untyped_fn = find_fn(fn_expr.id)
#      global_types =  
#      combine_args(untyped_fn.arg_names,
#        arg_types, 
#        untyped_fn.kwds, 
#        kwd_types, 
#        untyped_fn.global
#         
#      typed_fn = specialize(fn,   
#    
#    # if it's an operator  
#    # then create a typed operator node
#    # if it's a function then specialize it
#    # otherwise, throw an error since 
#    # only globally known functions are kosher...?
#    # what if it was defined in the local scope?
#    # def f(x):
#    #   def g(y):
#    #     return x + y
#    #   return g(3)
#    # called with f(2.0)
#    # ....
#    # - give f and g unique IDs like any other
#    #    variable
#    # - What about the bound closure variables 
#    #    of a function? 
#    # We can't just put "g" in a global lookup
#    # since it depends on references to variables
#    # of f's scope. 
#    # What if we include in a function's description
#    # the names of other variables it relies on
#    # and their types? 
#    # ...could a function then be returned? 
#    # def f(x):
#    #   def g(y):
#    #     return x + y
#    #   return g(3)
#    # What if we just do a simple closure conversion?
#    # CODE f(closure_f, x):
#    #   CODE g(closure_g, y):
#    #     x = closure_x[0]
#    #     return x + y
#    #   g = <package_fn g, x>
#    #   CALL(g, 3)
#    #   
#    # ...but this goes too far, since we're 
#    # giving functions first-class representation
#    # What if we instead just say:
#    #   g.code = "...", 
#    #   g.globals = "id1, id2, etc..."
#    #   and when typed
#    #   g.globals = "id1: t1, id2: t2"
#    # later we keep scoped  
#   
#   
#
#
#        