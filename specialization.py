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
    

import prims 

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
      return ptype.ClosureT(expr.fn, arg_types)
      
    def expr_Invoke():
      closure_t = expr_type(expr.closure)
      if isinstance(closure_t, ptype.ClosureT):
        closure_set = ptype.ClosureSet(closure_t)
      elif isinstance(closure_set, ptype.ClosureSet):
        closure_set = closure_t
      else:
        raise InferenceFailed("Invoke expected closure, but got %s" % closure_t)
      
      arg_types = map(expr_type, expr.args)
      invoke_result_type = ptype.Unknown
      for closure_type in closure_set.closures:
        untyped_id, closure_arg_types = closure_type.fn, closure_type.args
        untyped_fundef = untyped_functions[untyped_id]
        ret = infer_return_type(untyped_fundef, closure_arg_types + tuple(arg_types))
        invoke_result_type = invoke_result_type.combine(ret)
      return invoke_result_type
  
    def expr_PrimCall():
      arg_types = map(expr_type, expr.args)
      upcast_types = expr.prim.expected_input_types(arg_types)
      return expr.prim.result_type(upcast_types)
  
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
      assert isinstance(cond_type, ptype.ScalarT), \
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
  fn_return_type = old_type_env["$return"]
  
  var_map = {}
  new_type_env = {}
  for name in old_type_env.keys():
    assert isinstance(name, str), old_type_env 
  for (old_name, t) in old_type_env.iteritems():
    # don't try to rename '$return' 
    if not old_name.startswith("$"):
      new_name = names.refresh(old_name)
      var_map[old_name] = new_name
      new_type_env[new_name] = t
  
  def gen_temp(t, prefix = "temp"):
    temp = names.fresh(prefix)
    new_type_env[temp] = t
    return syntax.Var(temp, type = t)
      
  def typeof(expr):
    if isinstance(expr, syntax.Var):
      return new_type_env[expr.name]
    elif isinstance(expr, syntax.Tuple):
      return ptype.TupleT(map(typeof, expr.elts))
    elif isinstance(expr, syntax.Const):
      return ptype.type_of_value(expr.value)
    else:
      raise RuntimeError("Can't get type of %s" % expr)
    
  def rewrite_formal_arg(arg):
    # handle both the case when args are a flat list of strings
    # and a nested tree of expressions
    if isinstance(arg, str):
      return var_map[arg]
    elif isinstance(arg, syntax.Var):
      return syntax.Var(var_map[arg.name])
    elif isinstance(arg, syntax.Tuple):
      return syntax.Tuple(rewrite_formal_args(arg.elts))
  
  def rewrite_formal_args(args):
    return map(rewrite_formal_arg, args)
  
  def get_type(expr):
    return expr.type
  def get_types(exprs):
    return map(get_type, exprs)
  
  def rewrite_expr(expr):
    def rewrite_Var():
      old_name = expr.name
      new_name = var_map[old_name]
      var_type = new_type_env[new_name]
      return syntax.Var(new_name, type = var_type)
    
    def rewrite_Tuple():
      new_elts = map(rewrite_expr, expr.elts)
      new_types = get_types(new_elts)
      return syntax.Tuple(new_elts, type = ptype.TupleT(new_types))
    
    def rewrite_Const():
      return syntax.Const(expr.value, type = ptype.type_of_value(expr.value))
    
    def rewrite_PrimCall():
      # TODO: This awkwardly infers the types we need to cast args up to
      # but then dont' actually coerce them, since that's left as distinct work
      # for a later stage 
      new_args = map(rewrite_expr, expr.args)
      arg_types = map(get_type, new_args)
      upcast_types = expr.prim.expected_input_types(arg_types)
      result_type = expr.prim.result_type(upcast_types)
      upcast_args = [coerce_expr(x, t) for (x,t) in zip(new_args, upcast_types)]

      return syntax.PrimCall(expr.prim, upcast_args, type = result_type )
    def rewrite_Closure():
      new_args = map(rewrite_expr, expr.args)
      arg_types = map(get_type, new_args)
      closure_signature = ptype.ClosureT(fn = expr.fn, args = arg_types)
      return syntax.Closure(fn = expr.fn, args = new_args, type = closure_signature)
    
    def rewrite_Invoke():
      new_args = map(rewrite_expr, expr.args)
      arg_types = map(get_type, new_args)
      closure = rewrite_expr(expr.closure)
      if isinstance(closure.type, ptype.ClosureSet):
        closure_set = closure.type
      elif isinstance(closure.type, ptype.ClosureT):
        closure_set = ptype.ClosureSet(closure.type)
      else:
        raise InferenceFailed("Expected closure set, got %s" % expr.closure.type)
      return_type = ptype.Unknown
      for clos_sig in closure_set.closures:
        full_arg_types = clos_sig.args + tuple(arg_types)
        curr_return_type = infer_return_type(clos_sig.fn, full_arg_types)
        return_type = return_type.combine(curr_return_type)
      return syntax.Invoke(closure, new_args, type = return_type)
    
    return  dispatch(expr, 'rewrite')
  
  
  def cast(expr, t, curr_block = None):
    if curr_block is None:
      curr_block = blocks.current()
    assert isinstance(t, ptype.ScalarT), "Can't cast %s into %s" % (expr.type, t)  
    if hasattr(expr, 'name'):
      prefix = "%s.cast.%s" % (expr.name, t)
    else:
      prefix = "temp.cast.%s" % t
           
    temp = gen_temp(t, prefix = prefix) 
    cast =  syntax.Cast(expr, type = t)
    curr_block.append(syntax.Assign(temp, cast))
    return temp
  
  def coerce_expr(expr, t, curr_block = None):
    if expr.type is None:
      expr = rewrite_expr(expr)
      
    if expr.type == t:
      return expr
    elif isinstance(expr, syntax.Tuple):
      if not isinstance(t, ptype.TupleT) or len(expr.type.elt_types) != t.elt_types:
        raise ptype.IncompatibleTypes(expr.type, t)
      else:
        new_elts = []
        for elt, elt_t in zip(expr.elts, t.elt_types):
          new_elts.append(coerce_expr(elt, elt_t, curr_block))
        return syntax.Tuple(new_elts, type = t)
    else:
      return cast(expr, t, curr_block)
    
  def rewrite_merge(merge, left_block, right_block):
    typed_merge = {}
    for (old_var, (left, right)) in merge.iteritems():
      new_var = var_map[old_var]
      t = new_type_env[new_var]
      typed_left = coerce_expr(left, t, left_block)
      typed_right = coerce_expr(right, t, right_block)
      typed_merge[new_var] = (typed_left, typed_right) 
    return typed_merge 
  
  def rewrite_stmt(stmt):
    if isinstance(stmt, syntax.Assign):
      new_lhs = rewrite_expr(stmt.lhs)
      new_rhs = coerce_expr(stmt.rhs, new_lhs.type)
      return syntax.Assign(new_lhs, new_rhs)
    elif isinstance(stmt, syntax.If):
      new_cond = coerce_expr(stmt.cond, ptype.Bool)
      new_true_block = rewrite_block(stmt.true)
      new_false_block = rewrite_block(stmt.false)
      new_merge = rewrite_merge(stmt.merge, new_true_block, new_false_block)
      return syntax.If(new_cond, new_true_block, new_false_block, new_merge)
    elif isinstance(stmt, syntax.Return):
      return syntax.Return(coerce_expr(stmt.value, fn_return_type))
    elif isinstance(stmt, syntax.While):
      new_cond = coerce_expr(stmt.cond, ptype.Bool)
      new_body = rewrite_block(stmt.body)
      # insert coercions for left-branch values into the current block before
      # the while-loop and coercions for the right-branch to the end of the loop body
      new_merge_before = rewrite_merge(stmt.merge_before, 
        left_block = blocks.current(), right_block = new_body)
      new_merge_after = rewrite_merge(stmt.merge_after, 
        left_block = blocks.current(), right_block = new_body)
      return syntax.While(new_cond, new_body, new_merge_before, new_merge_after)
    else:
      raise RuntimeError("Not implemented: %s" % stmt)
    
    
  def rewrite_block(stmts):
    blocks.push()
    curr_block = blocks.current()
    for stmt in stmts:
      curr_block.append(rewrite_stmt(stmt))
    return blocks.pop()
  
  new_args = rewrite_formal_args(fn.args)
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
  typed_fundef = syntax.TypedFn(name = typed_id, args = new_args, 
    body = new_body, input_types = arg_types, return_type = fn_return_type, 
    type_env = new_type_env)
  return typed_fundef 


def specialize(untyped, arg_types): 
  if isinstance(untyped, str):
    untyped_id = untyped
    untyped = untyped_functions[untyped_id]
  else:
    assert isinstance(untyped, syntax.Fn)
    untyped_id = untyped.name 
  key = (untyped_id, tuple(arg_types))
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
      