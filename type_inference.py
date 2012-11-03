from collections import OrderedDict

import syntax as untyped_ast
import syntax as typed_ast


import core_types
import tuple_type
import array_type 
import closure_type 

import type_conv
import names 
from function_registry import untyped_functions, find_specialization, add_specialization
from common import dispatch
import args 
from syntax_helpers import get_type, get_types, unwrap_constant

import adverbs 
import adverb_helpers 
 

class InferenceFailed(Exception):
  def __init__(self, msg):
    self.msg = msg 



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


_invoke_type_cache = {}
def invoke_result_type(closure_t, arg_types):

  key = (closure_t, tuple(arg_types))
  if key in _invoke_type_cache:
    return _invoke_type_cache[key]
  else:
    if isinstance(closure_t, closure_type.ClosureT):
      closure_set = closure_type.ClosureSet(closure_t)
    elif isinstance(closure_set, closure_type.ClosureSet):
      closure_set = closure_t
    else:
      raise InferenceFailed("Invoke expected closure, but got %s" % closure_t)
      
    result_type = core_types.Unknown
    for closure_t in closure_set.closures:
      untyped_id, closure_arg_types = closure_t.fn, closure_t.args
      untyped_fundef = untyped_functions[untyped_id]
      ret = infer_return_type(untyped_fundef, closure_arg_types + tuple(arg_types))
      result_type = result_type.combine(ret)
    _invoke_type_cache[key] = result_type 
    return result_type 


def annotate_expr(expr, tenv, var_map):

  
  def annotate_child(child_expr):
    return annotate_expr(child_expr, tenv, var_map)
  
  def annotate_children(child_exprs):
    return [annotate_expr(e, tenv, var_map) for e in child_exprs]
  
  def expr_Closure():
    new_args = annotate_children(expr.args)
    t = closure_type.ClosureT(expr.fn, get_types(new_args))
    return typed_ast.Closure(expr.fn, new_args, type = t)
      
  def expr_Invoke():
    closure = annotate_child(expr.closure)
    args = annotate_children(expr.args)
    result_type = invoke_result_type(closure.type, get_types(args))
    return typed_ast.Invoke(closure, args, type = result_type) 
      
  def expr_Attribute():
    value = annotate_child(expr.value)
    assert isinstance(value.type, core_types.StructT)
    result_type = value.type.field_type(expr.name)
    return typed_ast.Attribute(value, expr.name, type = result_type)
  
  def expr_PrimCall():
    args = annotate_children(expr.args)
    arg_types = get_types(args)
    def get_elt_type(t):
      if isinstance(t, array_type.ArrayT):
        return t.elt_type
      else:
        return t
    def get_elt_types(ts):
      return map(get_elt_type, ts)
    
    if all([isinstance(t, core_types.ScalarT) for t in arg_types]):
      upcast_types = expr.prim.expected_input_types(arg_types)
      result_type = expr.prim.result_type(upcast_types)
      return typed_ast.PrimCall(expr.prim, args, type = result_type)
    else:
      scalar_arg_types = get_elt_types(arg_types)
      upcast_types = expr.prim.expected_input_types(scalar_arg_types)
      import prims
      prim_fn = prims.prim_wrapper(expr.prim)
      closure_t = closure_type.make_closure_type(prim_fn, [])
      
      scalar_result_type = invoke_result_type(closure_t, upcast_types)
      prim_closure = typed_ast.Closure(prim_fn, [], type = closure_t)
      max_rank = adverb_helpers.max_rank(arg_types)
      result_t = adverb_helpers.increase_rank(scalar_result_type, max_rank)
      return adverbs.Map(prim_closure, args, type = result_t)
  def expr_Index():
    value = annotate_child(expr.value)
    index = annotate_child(expr.index)
    if isinstance(value.type, tuple_type.TupleT):
      assert isinstance(index.type, core_types.IntT)
      assert isinstance(index, untyped_ast.Const)
      i = index.value
      assert isinstance(i, int)
      elt_t = value.type.elt_types[i]
      return typed_ast.TupleProj(value, i, type = elt_t)
    else:
      result_type = value.type.index_type(index.type)
      return typed_ast.Index(value, index, type = result_type)
  
  def expr_Array():
    new_elts = annotate_children(expr.elts)
    elt_types = get_types(new_elts)
    common_t = core_types.combine_type_list(elt_types)
    array_t = array_type.make_array_type(common_t, 1)
    return typed_ast.Array(new_elts, type = array_t)
  
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
  
  def expr_Map():
    closure = annotate_child(expr.fn)
    new_args = annotate_children(expr.args)
    
    arg_types = get_types(new_args)
    
    max_arg_rank = adverb_helpers.max_rank(arg_types)
     
    axis = unwrap_constant(expr.axis)
    n_outer_axes = 1 if (max_arg_rank > 0 and axis is not None) else max_arg_rank
    nested_types = adverb_helpers.lower_arg_ranks(arg_types, n_outer_axes)
    nested_result_type = invoke_result_type(closure.type, nested_types)
    result_type = adverb_helpers.increase_rank(nested_result_type, n_outer_axes)
    return adverbs.Map(fn = closure, args = new_args, axis = axis, type = result_type)
  
  def expr_Reduce():
    closure = annotate_child(expr.fn)
    new_args = annotate_children(expr.args)
    arg_types = get_types(new_args)
    max_arg_rank = adverb_helpers.max_rank(arg_types)
    axis = unwrap_constant(expr.axis)
    n_outer_axes = 1 if (max_arg_rank > 0 and axis is not None) else max_arg_rank
    nested_types = adverb_helpers.lower_arg_ranks(arg_types, n_outer_axes)
    nested_result_type = invoke_result_type(closure.type, nested_types)
    return adverbs.Reduce(fn = closure, args = new_args, axis = axis, type = nested_result_type)
  
  def expr_AllPairs():
    closure = annotate_child(expr.fn)
    new_args = annotate_children(expr.args)
    arg_types = get_types(new_args)
    
     
    axis = unwrap_constant(expr.axis)
    n_outer_axes = 2
    nested_types = adverb_helpers.lower_arg_ranks(arg_types, 1)
    nested_result_type = invoke_result_type(closure.type, nested_types)
    result_type = adverb_helpers.increase_rank(nested_result_type, n_outer_axes)
    return adverbs.AllPairs(fn = closure, args = new_args, axis = axis, type = result_type)
    
  result = dispatch(expr, prefix = "expr")
  assert result.type, "Missing type on %s" % result
  return result    



def annotate_stmt(stmt, tenv, var_map ):
  # print stmt
  
  
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
    for (var, values) in nodes.iteritems():
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
  
  def annotate_phi_nodes(nodes):
    new_nodes = {}
    for old_k, (old_left, old_right) in nodes.iteritems():
      new_name, (left, right) = annotate_phi_node(old_k, (old_left, old_right))
      new_nodes[new_name] = (left, right)
    return new_nodes 
  
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
      elif isinstance(lhs, untyped_ast.Index):
        new_arr = annotate_expr(lhs.value, tenv, var_map)
        new_idx = annotate_expr(lhs.index, tenv, var_map)
        
        assert isinstance(new_arr.type, array_type.ArrayT), "Expected array, got %s" % new_arr.type
        elt_t = new_arr.type.elt_type 
        return typed_ast.Index(new_arr, new_idx, type = elt_t)
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
    infer_left_flow(stmt.merge)
    cond = annotate_expr(stmt.cond, tenv, var_map)
    body = annotate_block(stmt.body, tenv, var_map)
    merge = annotate_phi_nodes(stmt.merge)
    return typed_ast.While(cond, body, merge)
    
  return dispatch(stmt, prefix="stmt")  

def annotate_block(stmts, tenv, var_map):
  return [annotate_stmt(s, tenv, var_map) for s in stmts]

def _infer_types(untyped_fn, positional_types, keyword_types = OrderedDict()):
  
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
  typed_args = untyped_fn.args.transform(var_map.rename)
  # flatten the positional, keyword, and default args into their
  # linear order, and use default_fn to get the type of default values
  input_types = typed_args.linearize_values(positional_types, keyword_types, default_fn = type_conv.typeof)
  tenv = {}
  args.match_list(typed_args.arg_slots, input_types, tenv)
  
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
    input_types = input_types, 
    return_type = return_type, 
    type_env = tenv)

  
  
  

def rewrite_typed(fn):
  
  type_env = fn.type_env
  def lookup(var):
    return type_env[var]
  
  fn_return_type = lookup("$return")
  
  def cast(expr, t):
    assert isinstance(t, core_types.ScalarT), "Casts not yet implemented for non-scalar types"
    return typed_ast.Cast(expr, type = t)
  
  def rewrite_expr(expr):
    """
    TODO: Make this recursive!
    """
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
    
    def rewrite_Array():
      array_t = expr.type
      elt_t = array_t.elt_type 
      new_elts = [coerce_expr(elt, elt_t) for elt in expr.elts]
      return typed_ast.Array(new_elts, type = array_t)
    
    return dispatch(expr, "rewrite", default = lambda x: x)
      
     
  def coerce_expr(expr, t):
    expr = rewrite_expr(expr)
    
    if expr.type == t:
      return expr
    
    elif isinstance(expr, untyped_ast.Tuple):
      if not isinstance(t, tuple_type.TupleT) or len(expr.type.elt_types) != t.elt_types:
        raise core_types.IncompatibleTypes(expr.type, t)
      else:
        new_elts = []
        for elt, elt_t in zip(expr.elts, t.elt_types):
          new_elts.append(coerce_expr(elt, elt_t))
        return typed_ast.Tuple(new_elts, type = t)
    else:
      return cast(expr, t)
    
  def rewrite_merge(merge):
    new_merge = {}
    for (var, (left, right)) in merge.iteritems():
      t = type_env[var]
      new_left = coerce_expr(left, t)
      new_right = coerce_expr(right, t)
      new_merge[var] = (new_left, new_right) 
    return new_merge 
  
  def rewrite_lhs(lhs):
    if isinstance(lhs, typed_ast.Var):
      t = lookup(lhs.name)
      if t == lhs.type:
        return lhs
      else:
        return typed_ast.Var(lhs.name, type = t)
    elif isinstance(lhs, typed_ast.Tuple):
      elts = map(rewrite_lhs, lhs.elts)
      elt_types = map(get_type, elts)
      if elt_types != lhs.type.elt_types:
        return typed_ast.Tuple(elts, type = tuple_type.make_tuple_type(elt_types))
      else:
        return lhs 
    else:
      return lhs 
      
  
  def rewrite_stmt(stmt):
    if isinstance(stmt, typed_ast.Assign):
      new_lhs = rewrite_lhs(stmt.lhs)
      lhs_t = new_lhs.type 
      new_rhs = coerce_expr(stmt.rhs, lhs_t)
      return typed_ast.Assign(stmt.lhs, new_rhs)
    
    elif isinstance(stmt, typed_ast.If):
      new_cond = coerce_expr(stmt.cond, core_types.Bool)
      new_true_block = rewrite_block(stmt.true)
      new_false_block = rewrite_block(stmt.false)
      new_merge = rewrite_merge(stmt.merge)
      return typed_ast.If(new_cond, new_true_block, new_false_block, new_merge)
    
    elif isinstance(stmt, untyped_ast.Return):
      return typed_ast.Return(coerce_expr(stmt.value, fn_return_type))
    
    
    elif isinstance(stmt, untyped_ast.While):
      new_cond = coerce_expr(stmt.cond, core_types.Bool)
      new_body = rewrite_block(stmt.body)
      # insert coercions for left-branch values into the current block before
      # the while-loop and coercions for the right-branch to the end of the loop body
      new_merge = rewrite_merge(stmt.merge)
      return typed_ast.While(new_cond, new_body, new_merge)
    else:
      raise RuntimeError("Not implemented: %s" % stmt)
    
    
  def rewrite_block(stmts):
    return map(rewrite_stmt, stmts)
  
  fn.body = rewrite_block(fn.body)
  

def specialize(untyped, arg_types): 
  if isinstance(untyped, str):
    untyped_id = untyped
    untyped = untyped_functions[untyped_id]
  else:
    assert isinstance(untyped, untyped_ast.Fn)
    untyped_id = untyped.name 
  
  try:
    return find_specialization(untyped_id, arg_types)
  except:
    typed_fundef = _infer_types(untyped, arg_types)
    rewrite_typed(typed_fundef)
    add_specialization(untyped_id, arg_types, typed_fundef)
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
      