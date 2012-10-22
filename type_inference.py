import syntax as untyped_ast
import syntax as typed_ast


import core_types
import tuple_type
from tuple_type import TupleT
import type_conv
import names 
from function_registry import untyped_functions, typed_functions
from common import dispatch
from args import Args, match, match_list, transform 


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



class VarMap:
  def __init__(self):
    self._vars = {}
    
  def rename(self, old_name):
    new_name = names.refresh(old_name)
    self._vars[old_name] = new_name
    return new_name
  
  def lookup(self, old_name):
    if old_name in self._vars:
      return self._vars[old_name]
    else:
      return self.rename(old_name)

def get_type(expr):
  return expr.type

def get_types(exprs):
  return [expr.type for expr in exprs]

def annotate_expr(expr, tenv, var_map):
  def annotate_child(child_expr):
    return annotate_expr(expr, tenv, var_map)
  
  def annotate_children(child_exprs):
    return [annotate_expr(e, tenv, var_map) for e in child_exprs]
  
  def expr_Closure():
    new_args = annotate_child(expr.args)
    t = core_types.ClosureT(expr.fn, get_types(new_args))
    return typed_ast.Closure(expr.fn, new_args, type = t)
      
  def expr_Invoke():
    closure = annotate_child(expr.closure)
    args = annotate_children(expr.args)
    closure_t = closure.type 
    if isinstance(closure_t, core_types.ClosureT):
      closure_set = core_types.ClosureSet(closure_t)
    elif isinstance(closure_set, core_types.ClosureSet):
      closure_set = closure_t
    else:
      raise InferenceFailed("Invoke expected closure, but got %s" % closure_t)
      
      arg_types = get_types(args)
      invoke_result_type = core_types.Unknown
      for closure_type in closure_set.closures:
        print invoke_result_type
        untyped_id, closure_arg_types = closure_type.fn, closure_type.args
        untyped_fundef = untyped_functions[untyped_id]
        ret = infer_return_type(untyped_fundef, closure_arg_types + tuple(arg_types))
        invoke_result_type = invoke_result_type.combine(ret)
    return typed_ast.Invoke(closure, args, type = invoke_result_type) 
      
  
  def expr_PrimCall():
    args = annotate_children(expr.args)
    arg_types = get_types(args)
    upcast_types = expr.prim.expected_input_types(arg_types)
    result_type = expr.prim.result_type(upcast_types)
    return typed_ast.PrimCall(expr.prim, args, type = result_type)
  
  def expr_Index():
    value = annotate_child(expr.value)
    index = annotate_child(expr.index)
    if isinstance(value.type, tuple_type.TupleT):
      assert isinstance(index.type, core_types.IntT)
      assert isinstance(index, untyped_ast.Const)
      i = index.value
      return typed_ast.TupleProj(value, i)
    else:
      result_type = value.type.index_type(index.type)
      return typed_ast.Index(value, index, type = result_type)
      
  def expr_Var():
    old_name = expr.name
    if old_name not in var_map._vars:
      raise names.NameNotFound(old_name)
    new_name = var_map.lookup(old_name)
    assert new_name in tenv 
    return typed_ast.Var(new_name, type = tenv[new_name])
    
  def expr_Tuple():
    elts = annotate_children(expr.elts)
    elt_types = get_types(elts)
    t = tuple_type.make_tuple_type(elt_types)
    return typed_ast.Tuple(elts, type = t)
  
  def expr_Const():
    return typed_ast.Const(expr.value, type_conv.typeof(expr.value))
  return dispatch(expr, prefix = "expr")
    



def annotate_stmt(stmt, tenv, var_map ):
  
  
  
  def infer_phi(result_var, val):
    """
    Don't actually rewrite the phi node, just 
    add any necessary types to the type environment
    """
    new_val = annotate_expr(val, tenv, var_map)
    new_type = new_val.type 
    old_type = tenv.get(result_var, core_types.Unknown)
    new_result_var = var_map.lookup(result_var)
    tenv[new_result_var]  = old_type.combine(new_type)
  
  def infer_phi_nodes(nodes, direction):
    for (var, values) in nodes:
      infer_phi(var, direction(values))
  
  def infer_left_flow(nodes):
    return infer_phi_nodes(nodes, lambda (x,_): x)
  
  def infer_right_flow(nodes):
    return infer_phi_nodes(nodes, lambda (_, x): x)
      
  
  def annotate_phi_node(result_var, (left_val, right_val)):
    """
    Rewrite the phi node by rewriting the values from either branch,
    renaming the result variable, recording its new type, 
    and returning the new name paired with the annotated branch values
     
    """  
    new_left = annotate_expr(left_val, tenv, var_map)
    new_right = annotate_expr(right_val, tenv, var_map)
    old_type = tenv.get(result_var, core_types.Unknown)
    new_type = old_type.combine(new_left.type).combine(new_right.type)
    new_var = var_map.lookup(result_var)
    tenv[new_var] = new_type
    return (new_var, (new_left, new_right))  
  
  def annotate_phi_nodes(nodes, flow_direction = None):
    return [annotate_phi_node(v, (l, r), flow_direction) for (v, (l,r)) in nodes]
  
  def stmt_Assign():
    rhs = annotate_expr(stmt.rhs, tenv, var_map)
    
    def annotate_lhs(lhs, rhs_type):
      if isinstance(lhs, untyped_ast.Tuple):
        assert isinstance(rhs_type, tuple_type.TupleT)
        assert len(lhs.elts) == len(rhs_type.elt_types)
        new_elts = [annotate_lhs(elt, elt_type) for (elt, elt_type) in 
                    zip(lhs.elts, rhs_type.elt_types)]
        tuple_t = tuple_type.make_tuple_type(get_types(new_elts))
        return typed_ast.Tuple(new_elts, type = tuple_t)
      else:
        assert isinstance(lhs, untyped_ast.Var)
        new_name = var_map.lookup(lhs.name)
        old_type = tenv.get(new_name, core_types.Unknown)
        new_type = old_type.combine(rhs_type)
        tenv[new_name] = new_type
        return typed_ast.Var(new_name, type = new_type)
      
    lhs = annotate_lhs(stmt.lhs, rhs.type)
    return typed_ast.Assign(lhs, rhs)

  def stmt_If():
    cond = annotate_expr(stmt.cond, tenv, var_map)
    assert isinstance(cond.type, core_types.ScalarT), \
      "Condition has type %s but must be convertible to bool" % cond.type
    true = annotate_block(stmt.true, tenv, var_map)
    false = annotate_block(stmt.false, tenv, var_map)
    merge = annotate_phi_nodes(stmt.merge)
    return typed_ast.If(cond, true, false, merge) 
   
  def stmt_Return():
    ret_val = annotate_expr(stmt.value, tenv, var_map)
    curr_return_type = tenv["$return"]
    tenv["$return"] = curr_return_type.combine(ret_val.type)
    return typed_ast.Return(ret_val)
    
  def stmt_While():
    infer_left_flow(stmt.merge_before)
    cond = annotate_expr(stmt.cond, tenv, var_map) 
    body = annotate_block(stmt.body)
    merge_before = annotate_phi_nodes(stmt.merge_before)
    merge_after = annotate_phi_nodes(stmt.merge_after)
    return typed_ast.While(cond, body, merge_before, merge_after)
    
  return dispatch(stmt, prefix="stmt")  

def annotate_block(stmts, tenv, var_map):
  return [annotate_stmt(s, tenv, var_map) for s in stmts]

def _infer_types(untyped_fn, arg_types):
  """
  Given an untyped function and input types, 
  propagate the types through the body, 
  annotating the AST with type annotations.
   
  NOTE: The AST won't be in a correct state
  until a rewrite pass back-propagates inferred 
  types throughout the program and inserts
  adverbs for scalar operators applied to arrays
  """
  
  
  var_map = VarMap()
  
  arg_patterns = untyped_fn.args.positional + untyped_fn.args.kwds.keys() 
  typed_args = untyped_fn.args.transform(var_map.rename)

  
  assert isinstance(typed_args, Args)
  tenv = typed_args.bind(typed_args, arg_types)
  
  # keep track of the return 
  tenv['$return'] = core_types.Unknown 
  
  body = annotate_block(untyped_fn.body, tenv, var_map)
  return_type = tenv['$return']
  # if nothing ever gets returned, then set the return type to None
  if return_type == core_types.Unknown:
    assert False, "TO DO: Implement a none type"
    
  return typed_ast.TypedFn(
    name = names.refresh(untyped_fn.name), 
    body = body, 
    args = typed_args, 
    input_types = arg_types, 
    return_type = return_type, 
    type_env = tenv)

  
  
  

def rewrite_typed(fn):
  print fn
  blocks = NestedBlocks()
  old_type_env = fn.type_env
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
    return typed_ast.Var(temp, type = t)
      
  def typeof(expr):
    if isinstance(expr, untyped_ast.Var):
      return new_type_env[expr.name]
    elif isinstance(expr, untyped_ast.Tuple):
      return tuple_type.TupleT(map(typeof, expr.elts))
    elif isinstance(expr, untyped_ast.Const):
      return type_conv.typeof(expr.value)
    else:
      raise RuntimeError("Can't get type of %s" % expr)
    
  def rewrite_formal_arg(arg):
    # handle both the case when args are a flat list of strings
    # and a nested tree of expressions
    if isinstance(arg, str):
      return var_map[arg]
    elif isinstance(arg, untyped_ast.Var):
      return typed_ast.Var(var_map[arg.name])
    elif isinstance(arg, untyped_ast.Tuple):
      return typed_ast.Tuple(rewrite_formal_args(arg.elts))
  
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
      return typed_ast.Var(new_name, type = var_type)
    
    def rewrite_Tuple():
      
      new_elts = map(rewrite_expr, expr.elts)
      new_types = get_types(new_elts)
      print expr, new_elts, new_types
      return typed_ast.Tuple(new_elts, type = tuple_type.make_tuple_type(new_types))
    
    def rewrite_Const():
      return typed_ast.Const(expr.value, type = type_conv.typeof(expr.value))
    
    def rewrite_Index():
      ###
      ### HOW DO TO THIS WITHOUT DUPLICATING THE CHECK AGAINST Type.static_indexing
      assert False 
    
    def rewrite_PrimCall():
      # TODO: This awkwardly infers the types we need to cast args up to
      # but then dont' actually coerce them, since that's left as distinct work
      # for a later stage 
      new_args = map(rewrite_expr, expr.args)
      arg_types = map(get_type, new_args)
      upcast_types = expr.prim.expected_input_types(arg_types)
      result_type = expr.prim.result_type(upcast_types)
      upcast_args = [coerce_expr(x, t) for (x,t) in zip(new_args, upcast_types)]

      return typed_ast.PrimCall(expr.prim, upcast_args, type = result_type )
    def rewrite_Closure():
      new_args = map(rewrite_expr, expr.args)
      arg_types = map(get_type, new_args)
      closure_signature = core_types.ClosureT(fn = expr.fn, args = arg_types)
      return typed_ast.Closure(fn = expr.fn, args = new_args, type = closure_signature)
    
    def rewrite_Invoke():
      new_args = map(rewrite_expr, expr.args)
      arg_types = map(get_type, new_args)
      closure = rewrite_expr(expr.closure)
      if isinstance(closure.type, core_types.ClosureSet):
        closure_set = closure.type
      elif isinstance(closure.type, core_types.ClosureT):
        closure_set = core_types.ClosureSet(closure.type)
      else:
        raise InferenceFailed("Expected closure set, got %s" % expr.closure.type)
      return_type = core_types.Unknown
      for clos_sig in closure_set.closures:
        full_arg_types = clos_sig.args + tuple(arg_types)
        curr_return_type = infer_return_type(clos_sig.fn, full_arg_types)
        return_type = return_type.combine(curr_return_type)
      return typed_ast.Invoke(closure, new_args, type = return_type)
    
    return  dispatch(expr, 'rewrite')
  
  
  def cast(expr, t, curr_block = None):
    if curr_block is None:
      curr_block = blocks.current()
    assert isinstance(t, core_types.ScalarT), "Can't cast %s into %s" % (expr.type, t)  
    if hasattr(expr, 'name'):
      prefix = "%s.cast.%s" % (expr.name, t)
    else:
      prefix = "temp.cast.%s" % t
           
    temp = gen_temp(t, prefix = prefix) 
    cast =  typed_ast.Cast(expr, type = t)
    curr_block.append(typed_ast.Assign(temp, cast))
    return temp
  
  def coerce_expr(expr, t, curr_block = None):
    if expr.type is None:
      expr = rewrite_expr(expr)
      
    if expr.type == t:
      return expr
    
    elif isinstance(expr, untyped_ast.Tuple):
      if not isinstance(t, tuple_type.TupleT) or len(expr.type.elt_types) != t.elt_types:
        raise core_types.IncompatibleTypes(expr.type, t)
      else:
        new_elts = []
        for elt, elt_t in zip(expr.elts, t.elt_types):
          new_elts.append(coerce_expr(elt, elt_t, curr_block))
        return typed_ast.Tuple(new_elts, type = t)
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
    if isinstance(stmt, untyped_ast.Assign):
      new_lhs = rewrite_expr(stmt.lhs)
      new_rhs = coerce_expr(stmt.rhs, new_lhs.type)
      return typed_ast.Assign(new_lhs, new_rhs)
    elif isinstance(stmt, untyped_ast.If):
      new_cond = coerce_expr(stmt.cond, core_types.Bool)
      new_true_block = rewrite_block(stmt.true)
      new_false_block = rewrite_block(stmt.false)
      new_merge = rewrite_merge(stmt.merge, new_true_block, new_false_block)
      return typed_ast.If(new_cond, new_true_block, new_false_block, new_merge)
    elif isinstance(stmt, untyped_ast.Return):
      return typed_ast.Return(coerce_expr(stmt.value, fn_return_type))
    
    
    elif isinstance(stmt, untyped_ast.While):
      new_cond = coerce_expr(stmt.cond, core_types.Bool)
      new_body = rewrite_block(stmt.body)
      # insert coercions for left-branch values into the current block before
      # the while-loop and coercions for the right-branch to the end of the loop body
      new_merge_before = rewrite_merge(stmt.merge_before, 
        left_block = blocks.current(), right_block = new_body)
      new_merge_after = rewrite_merge(stmt.merge_after, 
        left_block = blocks.current(), right_block = new_body)
      return typed_ast.While(new_cond, new_body, new_merge_before, new_merge_after)
    else:
      raise RuntimeError("Not implemented: %s" % stmt)
    
    
  def rewrite_block(stmts):
    blocks.push()
    curr_block = blocks.current()
    for stmt in stmts:
      curr_block.append(rewrite_stmt(stmt))
    return blocks.pop()
  
  fn.body = rewrite_block(fn.body)
  

def specialize(untyped, arg_types): 
  if isinstance(untyped, str):
    untyped_id = untyped
    untyped = untyped_functions[untyped_id]
  else:
    assert isinstance(untyped, untyped_ast.Fn)
    untyped_id = untyped.name 
  key = (untyped_id, tuple(arg_types))
  if key in typed_functions:
    return typed_functions[key]
  else:
    typed_fundef = _infer_types(untyped, arg_types)  
    rewrite_typed(typed_fundef)

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
      