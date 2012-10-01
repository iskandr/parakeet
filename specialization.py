import syntax
import ptype
import names 
from function_registry import untyped_functions, typed_functions
from common import dispatch
from match import match, match_list

   

class InferenceFailed(Exception):
  def __init__(self, msg):
    self.msg = msg 


class NestedBlocks:
  def __init__(self):
    self.blocks = []
  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    return self.blocks.pop()
  
  def current(self):
    return self.blocks[-1]
  
  def append_to_current(self, stmt):
    self.current().append(stmt)
  
  def extend_current(self, stmts):
    self.current().extend(stmts)
    
     

def _infer_types(fn, arg_types):
  """
  Actual implementation of type inference which doesn't attempt to 
  look up cached version of typed function
  """ 
  tenv = match_list(fn.args, arg_types)
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
        ret = infer_return_type(untyped_fundef, closure_arg_types + arg_types)
        invoke_result_type = invoke_result_type.combine(ret)
      return invoke_result_type
  
    def expr_PrimCall():
      arg_types = map(expr_type, expr.args)
      # TODO: make this actually figure out the return type of a ufunc 
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
  
  def merge_left_branch(phi_nodes):
    for result_var, (left_val, _) in phi_nodes.iteritems():
      left_type = expr_type(left_val)
      old_type = tenv.get(result_var, ptype.Unknown)
      tenv[result_var] = old_type.combine(left_type)
  
  def merge_right_branch(phi_nodes):
    for result_var, (_, right_val) in phi_nodes.iteritems():
      right_type = expr_type(right_val)
      old_type = tenv.get(result_var, ptype.Unknown)
      tenv[result_var] = old_type.combine(right_type)
  
      
  def merge_branches(phi_nodes):
    for result_var, (left_val, right_val) in phi_nodes.iteritems():
      left_type = expr_type(left_val)
      right_type = expr_type(right_val)
      old_type = tenv.get(result_var, ptype.Unknown)
      tenv[result_var]  = old_type.combine(left_type).combine(right_type)
  
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
    
    def stmt_While():
      merge_left_branch(stmt.merge_before)
      analyze_block(stmt.body)
      merge_right_branch(stmt.merge_after)
      merge_branches(stmt.merge_after)
    
    dispatch(stmt, prefix="stmt")
    
    
  def analyze_block(stmts):
    for stmt in stmts:
      analyze_stmt(stmt)
  analyze_block(fn.body)
  return tenv
  

def rewrite_typed(fn, old_type_env):
  
  blocks = NestedBlocks()
  
  var_map = {}
  new_type_env = {}
  for (old_name, t) in old_type_env.iteritems():
    # don't try to rename '$return' 
    if not old_name.startswith("$"):
      new_name = names.refresh(old_name)
      var_map[old_name] = new_name
      new_type_env[new_name] = t
  
  def typeof(expr):
    if isinstance(expr, syntax.Var):
      return new_type_env[expr.name]
    elif isinstance(expr, syntax.Tuple):
      return ptype.Tuple(map(typeof, expr.elts))
    elif isinstance(expr, syntax.Const):
      return ptype.type_of_value(expr.value)
    else:
      raise RuntimeError("Can't get type of %s" % expr)
    
  def rewrite_arg(arg):
    # handle both the case when args are a flat list of strings
    # and a nested tree of expressions
    if isinstance(arg, str):
      return var_map[arg]
    elif isinstance(arg, syntax.Var):
      return syntax.Var(var_map[arg.name])
    elif isinstance(arg, syntax.Tuple):
      return syntax.Tuple(rewrite_args(arg.elts))
  
  def rewrite_args(args):
    return map(rewrite_arg, args)
  
  def tag(expr, t):
    expr.type = t 
    return expr 
  
  def rewrite_expr(expr):
    def rewrite_Var():
      old_name = expr.name
      new_name = var_map[old_name]
      var_type = new_type_env[new_name]
      return syntax.Var(new_name), var_type
    
    def rewrite_Tuple():
      new_elts = map(rewrite_expr, expr.elts)
      new_types = map(lambda e: e.type, new_elts)
      return syntax.Tuple(new_elts), ptype.Tuple(new_types)
      
    new_expr, expr_type = dispatch(expr, 'rewrite')
    new_expr.type = expr_type
    return new_expr 
    
  
  def cast(expr, t):
    return None
  
  def coerce_expr(expr, t):
    if expr.type == t:
      return expr
    def coerce_Tuple():
      if not isinstance(t, ptype.Tuple) or len(expr.type.elt_types) != t.elt_types:
        raise ptype.IncompatibleTypes(expr.type, t)
      else:
        new_elts = []
        for elt, elt_t in zip(expr.elts, t.elt_types):
          new_elts.append(coerce_expr(elt, elt_t))
        new_tuple = syntax.Tuple(new_elts)
        new_tuple.type = t 
        return new_tuple
    def coerce_Var():
      assert isinstance(t, ptype.Scalar), "Can't cast %s into %s" % (expr.type, t)       
      
    def coerce_Var():
       
    curr_t = typeof(expr)
    if curr_t == t:
      return 
    
  def rewrite_stmt(stmt):
    if isinstance(stmt, syntax.Assign):
      new_lhs = rewrite_expr(stmt.lhs)
      expected_type = new_lhs.type
      
      new_rhs = coerce_expr(rewrite_expr(stmt.rhs), expected_type)
      return syntax.Assign(new_lhs, new_rhs)
    
    
  def rewrite_block(stmts):
    blocks.push()
    curr_block = blocks.current()
    for stmt in stmts:
      curr_block.append(rewrite_stmt(stmt))
    return blocks.pop()
  
  new_args = rewrite_args(fn.args)
  new_body = rewrite_block(fn.body)
  
  typed_id = names.refresh(fn.name)
  # this helper only exists since args are currently either strings or expressions
  # TODO: make args always expressions 
  def arg_type(arg):
    if isinstance(arg, str):
      return new_type_env[arg]
    else:
      return typeof(arg)
     
  arg_types = map(arg_type, new_args)
  return_type = old_type_env["$return"]
  typed_fundef = syntax.TypedFn(name = typed_id, args = new_args, 
    body = new_body, input_types = arg_types, return_type = return_type, 
    type_env = new_type_env)
  return typed_fundef 


def specialize(untyped, arg_types): 
  key = (untyped.name, tuple(arg_types))
  if key in typed_functions:
    return typed_functions[key]
  else:
    type_env = _infer_types(untyped, arg_types)  
    typed_fundef = rewrite_typed(untyped, type_env)

    typed_functions[key] = typed_fundef 
    return typed_fundef 
 
def infer_return_type(untyped, arg_types):
  """
  Given a function definition and some input types, 
  gives back the return type 
  and implicitly generates a specialized version of the
  function. 
  """
  typed = specialize(untyped, arg_types)
  return typed.return_type 
      