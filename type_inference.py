from collections import OrderedDict

import syntax as untyped_ast
import syntax as typed_ast
import syntax_helpers 

import core_types
import tuple_type
import array_type 
import closure_type 

import type_conv
import names 
from function_registry import untyped_functions, find_specialization, \
                              add_specialization
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

def get_invoke_specialization(closure_t, arg_types):
  # for a given closure and the direct argument types it
  # receives when invokes, return the specialization which 
  # will ultimately get called 
  if isinstance(arg_types, list):
    arg_types = tuple(arg_types)
  # for a given invocation of a ClosureT
  # what is the typed_fn that gets called? 
  untyped_id, closure_arg_types = closure_t.fn, closure_t.arg_types
  untyped_fundef = untyped_functions[untyped_id]
  full_arg_types = closure_arg_types + arg_types 
  return specialize(untyped_fundef, full_arg_types)
   

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
      typed_fundef = get_invoke_specialization(closure_t, arg_types)
      result_type = result_type.combine(typed_fundef.return_type)
    _invoke_type_cache[key] = result_type 
    return result_type 
  
def annotate_expr(expr, tenv, var_map):
  print "expr", expr 
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
    
    if all(isinstance(t, core_types.ScalarT) for t in arg_types):
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
      result_t = array_type.increase_rank(scalar_result_type, max_rank)
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
    array_t = array_type.increase_rank(common_t, 1)
    return typed_ast.Array(new_elts, type = array_t)
  
  def expr_Slice():
    start = annotate_child(expr.start)
    stop = annotate_child(expr.stop)
    step = annotate_child(expr.step)
    slice_t = array_type.make_slice_type(start.type, stop.type, step.type)
    return typed_ast.Slice(start, stop, step, type = slice_t)

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
    axis = unwrap_constant(expr.axis)
    arg_types = get_types(new_args)
    result_type = infer_map_type(closure.type, arg_types, axis)
    if axis is None and adverb_helpers.max_rank(arg_types) == 1:
      axis = 0
    return adverbs.Map(fn = closure, args = new_args, axis = axis, type = result_type)
  
  def expr_Reduce():
    closure = annotate_child(expr.fn)
    new_args = annotate_children(expr.args)
    arg_types = get_types(new_args)
    axis = unwrap_constant(expr.axis)
    result_type = infer_reduce_type(closure.type, arg_types, axis, None, None)
    if axis is None and adverb_helpers.max_rank(arg_types) == 1:
      axis = 0
    return adverbs.Reduce(fn = closure, 
                          args = new_args, 
                          axis = axis, 
                          type = result_type)
  
  def expr_AllPairs():
    closure = annotate_child(expr.fn)
    new_args = annotate_children(expr.args)
    arg_types = get_types(new_args)
    assert len(arg_types) == 2
    axis = unwrap_constant(expr.axis)
    result_type = infer_allpairs_type(closure.type, arg_types[0], arg_types[1], axis)
    if axis is None and adverb_helpers.max_rank(arg_types) == 1:
      axis = 0
    return adverbs.AllPairs(fn = closure, 
                            args = new_args, 
                            axis = axis, 
                            type = result_type)
    
  result = dispatch(expr, prefix = "expr")
  assert result.type, "Missing type on %s" % result
  assert isinstance(result.type, core_types.Type), \
    "Unexpected type annotation on %s: %s" % (expr, result.type)
  return result    

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
      elif isinstance(lhs, untyped_ast.Attribute):
        name = lhs.name 
        struct = annotate_expr(lhs.value, tenv, var_map)
        struct_t = struct.type 
        assert isinstance(struct_t, core_types.StructT), \
          "Can't access fields on value %s of type %s" % \
          (struct, struct_t)
        field_t = struct_t.field_type(name)
        return typed_ast.Attribute(struct, name, field_t)
      else:
        assert isinstance(lhs, untyped_ast.Var), \
          "Unexpected LHS: " + str(lhs)
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
  
  tenv = typed_args.bind(positional_types, keyword_types, 
                         default_fn = type_conv.typeof,
                         varargs_fn = tuple_type.make_tuple_type)
  
  input_types = [tenv[arg_name] for arg_name in typed_args.arg_slots]
  if typed_args.varargs:
    varargs_tuple_t = tenv[typed_args.varargs]
    input_types += varargs_tuple_t.elt_types 
  
  # keep track of the return 
  tenv['$return'] = core_types.Unknown 
  
  body = annotate_block(untyped_fn.body, tenv, var_map)
  return_type = tenv["$return"]
  # if nothing ever gets returned, then set the return type to None
  if isinstance(return_type,  core_types.UnknownT):

    body.append(typed_ast.Return(syntax_helpers.const_none))
    tenv["$return"] = core_types.NoneType
    return_type = core_types.NoneType
    
  return typed_ast.TypedFn(
    name = names.refresh(untyped_fn.name), 
    body = body, 
    args = typed_args, 
    input_types = input_types, 
    return_type = return_type, 
    type_env = tenv)


from insert_coercions import insert_coercions 

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
    coerced_fundef = insert_coercions(typed_fundef) 
    
    import optimize 
    # TODO: Also store the unoptimized version 
    # so we can do adaptive recompilation  
    opt = optimize.optimize(coerced_fundef, copy = False)
    add_specialization(untyped_id, arg_types, opt)
    return opt 

def infer_return_type(untyped, arg_types):
  """
  Given a function definition and some input types, 
  gives back the return type 
  and implicitly generates a specialized version of the
  function. 
  """
  # print "Specializing for %s: %s" % (arg_types, untyped )
  typed = specialize(untyped, arg_types)
  return typed.return_type 


def infer_reduce_type(closure_t, arg_types, axis, init = None, combine = None):
  if init is None:
    #
    #The simplest reductions assume the initial value, 
    #the carried accumulator, and the element type
    #are all the same (and there's only one element type
    #since there's only one array input
    #
    assert len(arg_types) == 1
    input_type = arg_types[0]
    n_outer_axes = adverb_helpers.num_outer_axes(arg_types, axis)
    nested_type = array_type.lower_rank(input_type, n_outer_axes)
    nested_result_type = invoke_result_type(closure_t, [nested_type, nested_type])
    assert nested_type == nested_result_type, \
      "Can't yet handle accumulator type %s which differs from input %s" % \
      (nested_result_type, nested_type)
    
    if combine is not None:
      combine_result_t = invoke_result_type(combine, [nested_type, nested_type])
      assert combine_result_t == nested_type, \
        "Wrong type for combiner result, expected %s but got %s" % \
        (nested_type, combine_result_t)
    return nested_result_type 
  else:
    raise RuntimeError("Type inference not implemented for complex reductions")



def infer_scan_type(closure_t, arg_types, axis, init = None, combine = None):
  n_outer_axes = adverb_helpers.num_outer_axes(arg_types, axis)
  acc_t = infer_reduce_type(closure_t, arg_types, axis, init, combine)
  return array_type.increase_rank(acc_t, n_outer_axes) 

def infer_map_type(closure_t, arg_types, axis):
  print "infer_map_type"
  print "-- closure", closure_t 
  print "-- arg_types", arg_types 
  n_outer_axes = adverb_helpers.num_outer_axes(arg_types, axis)
  nested_types = array_type.lower_ranks(arg_types, n_outer_axes)
  nested_result_type = invoke_result_type(closure_t, nested_types)
  return array_type.increase_rank(nested_result_type, n_outer_axes)

def infer_allpairs_type(closure_t, xtype, ytype, axis):
  axis = unwrap_constant(axis)
  n_outer_axes = 2
  arg_types = [xtype, ytype]
  if axis is None:
    nested_types = array_type.elt_types(arg_types)
  else:
    nested_types = array_type.lower_ranks(arg_types, 1)
  nested_result_type = invoke_result_type(closure_t, nested_types)
  return array_type.increase_rank(nested_result_type, n_outer_axes)
  
  