import adverbs
import adverb_helpers
import adverb_wrapper
import array_type
import closure_type
import config
import names
import prims
import syntax as untyped_ast
import syntax as typed_ast
import syntax_helpers
import tuple_type
import type_conv

from args import ActualArgs
from array_type import ArrayT
from common import dispatch
from core_types import Type, IntT, Int64,  ScalarT
from core_types import NoneType, NoneT, Unknown, UnknownT
from core_types import combine_type_list, StructT
from syntax_helpers import get_type, get_types, unwrap_constant
from syntax_helpers import one_i64, zero_i64, none
from tuple_type import TupleT, make_tuple_type
from stride_specialization import specialize

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

  def __str__(self):
    return "VarMap(%s)" % self._vars

def unpack_closure(closure):
  """
  Given an object which could be either a function, a function's name, a
  closure, or a closure type:
  Return the underlying untyped function and the closure arguments
  """

  if closure.__class__ is closure_type.ClosureT:
    fn, closure_args = closure.fn, closure.arg_types

  elif closure.type.__class__ is closure_type.ClosureT:
    fn, arg_types = closure.type.fn, closure.type.arg_types
    closure_args = \
        [typed_ast.ClosureElt(closure, i, type = arg_t)
         for (i, arg_t) in enumerate(arg_types)]
  else:
    fn = closure
    closure_args = []
  if isinstance(fn, str):
    fn = untyped_ast.Fn.registry[fn]
  return fn, closure_args

def make_typed_closure(untyped_closure, typed_fn):
  if untyped_closure.__class__ is untyped_ast.Fn:
    return typed_fn

  assert isinstance(untyped_closure, untyped_ast.Expr) and \
      isinstance(untyped_closure.type, closure_type.ClosureT)
  _, closure_args = unpack_closure(untyped_closure)
  if len(closure_args) == 0:
    return typed_fn
  else:
    t = closure_type.make_closure_type(typed_fn, get_types(closure_args))
    return typed_ast.Closure(typed_fn, closure_args, t)

def linearize_arg_types(fn, args):
  """
  Given a function object which might be one of:
    (1) a closure type
    (2) the name of an untyped function
    (3) an untyped fn object
  and some argument types which might
    (1) a list
    (2) a tuple
    (3) an ActualArgs object
  linearize the argument types with respect to
  the untyped function's argument order and return
  both the untyped function and list of arguments
  """

  untyped_fundef, closure_args = unpack_closure(fn)
  if isinstance(args, (list, tuple)):
    args = ActualArgs(args)

  if len(closure_args) > 0:
    args = args.prepend_positional(closure_args)

  def keyword_fn(_, v):
    return type_conv.typeof(v)

  linear_args, extra = \
      untyped_fundef.args.linearize_values(args, keyword_fn = keyword_fn)
  return untyped_fundef, tuple(linear_args + extra)

def tuple_elts(tup):
  return [typed_ast.TupleProj(tup, i, t)
          for (i,t) in enumerate(tup.type.elt_types)]

def flatten_actual_args(args):
  if isinstance(args, (list,tuple)):
    return args
  assert isinstance(args, ActualArgs), \
      "Unexpected args: %s" % (args,)
  assert len(args.keywords) == 0
  result = list(args.positional)
  if args.starargs:
    result.extend(tuple_elts(args.starargs))
  return result

def linearize_actual_args(fn, args):
    untyped_fn, closure_args = unpack_closure(fn)

    if isinstance(args, (list, tuple)):
      args = ActualArgs(args)
    args = args.prepend_positional(closure_args)

    arg_types = args.transform(syntax_helpers.get_type)

    # Drop arguments that are assigned defaults,
    # since we're assuming those are set in the body
    # of the function
    combined_args = untyped_fn.args.linearize_without_defaults(args, tuple_elts)
    return untyped_fn, combined_args, arg_types

_invoke_type_cache = {}
def invoke_result_type(fn, arg_types):
  if isinstance(fn, typed_ast.TypedFn):
    assert isinstance(arg_types, (list, tuple))
    assert len(arg_types) == len(fn.input_types), \
        "Type mismatch between expected inputs %s and %s" % \
        (fn.input_types, arg_types)
    assert all(t1 == t2 for (t1,t2) in zip(arg_types, fn.input_types))
    return fn.return_type

  if isinstance(arg_types, (list, tuple)):
    arg_types = ActualArgs(arg_types)

  key = (fn, arg_types)
  if key in _invoke_type_cache:
    return _invoke_type_cache[key]

  if isinstance(fn, closure_type.ClosureT):
    closure_set = closure_type.ClosureSet(fn)
  else:
    assert isinstance(fn, closure_type.ClosureSet), \
        "Invoke expected closure, but got %s" % (fn,)
    closure_set = fn

  result_type = Unknown
  for closure_t in closure_set.closures:
    typed_fundef = specialize(closure_t, arg_types)
    result_type = result_type.combine(typed_fundef.return_type)
  _invoke_type_cache[key] = result_type
  return result_type

def annotate_expr(expr, tenv, var_map):
  def annotate_child(child_expr):
    return annotate_expr(child_expr, tenv, var_map)

  def annotate_children(child_exprs):
    return [annotate_expr(e, tenv, var_map) for e in child_exprs]

  def annotate_args(args, flat = False):
    if isinstance(args, (list, tuple)):
      return map(annotate_child, args)
    else:
      new_args = args.transform(annotate_child)
      if flat:
        return flatten_actual_args(new_args)
      else:
        return new_args

  def annotate_keywords(kwds_dict):
    if kwds_dict is None:
      return {}
    else:
      result = {}
      for (k,v) in kwds_dict.iteritems():
        result[k] = annotate_child(v)
      return result

  def keyword_types(kwds):
    keyword_types = {}
    for (k,v) in kwds.iteritems():
      keyword_types[k] = get_type(v)
    return keyword_types

  def expr_Closure():
    new_args = annotate_children(expr.args)
    t = closure_type.make_closure_type(expr.fn, get_types(new_args))
    return typed_ast.Closure(expr.fn, new_args, type = t)

  def prim_to_closure(p):
    untyped_fn = prims.prim_wrapper(p)
    t = closure_type.make_closure_type(untyped_fn, ())
    return typed_ast.Closure(untyped_fn, (), type = t)

  def expr_Arith():
    return prim_to_closure(expr)

  def expr_Fn():
    t = closure_type.make_closure_type(expr, ())
    return typed_ast.Closure(expr, [], type = t)

  def expr_Call():
    closure = annotate_child(expr.fn)
    args = annotate_args(expr.args)
    untyped_fn, args, arg_types = linearize_actual_args(closure, args)

    typed_fn = specialize(untyped_fn, arg_types)
    return typed_ast.Call(typed_fn, args, typed_fn.return_type)

  def expr_Attribute():
    value = annotate_child(expr.value)
    assert isinstance(value.type, StructT)
    result_type = value.type.field_type(expr.name)
    return typed_ast.Attribute(value, expr.name, type = result_type)

  def expr_PrimCall():
    args = annotate_args(expr.args)
    arg_types = get_types(args)

    if all(isinstance(t, ScalarT) for t in arg_types):
      upcast_types = expr.prim.expected_input_types(arg_types)
      result_type = expr.prim.result_type(upcast_types)
      return typed_ast.PrimCall(expr.prim, args, type = result_type)
    else:
      assert all(not isinstance(t, NoneT) for t in arg_types)
      prim_fn = prims.prim_wrapper(expr.prim)

      max_rank = adverb_helpers.max_rank(arg_types)
      arg_names = adverb_wrapper.gen_data_arg_names(len(arg_types))
      untyped_broadcast_fn = \
          adverb_helpers.nested_maps(prim_fn, max_rank, arg_names)
      typed_broadcast_fn = specialize(untyped_broadcast_fn, arg_types)
      result_t = typed_broadcast_fn.return_type
      return typed_ast.Call(typed_broadcast_fn, args, type = result_t)

  def expr_Index():
    value = annotate_child(expr.value)
    index = annotate_child(expr.index)
    if isinstance(value.type, TupleT):
      assert isinstance(index.type, IntT)
      assert isinstance(index, untyped_ast.Const)
      i = index.value
      assert isinstance(i, int)
      elt_types = value.type.elt_types
      assert i < len(elt_types), \
          "Can't get element %d of length %d tuple %s : %s" % \
          (i, len(elt_types), value, value.type)
      elt_t = value.type.elt_types[i]
      return typed_ast.TupleProj(value, i, type = elt_t)
    else:
      result_type = value.type.index_type(index.type)
      return typed_ast.Index(value, index, type = result_type)

  def expr_Array():
    new_elts = annotate_args(expr.elts)
    elt_types = get_types(new_elts)
    common_t = combine_type_list(elt_types)
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
    assert new_name in tenv, \
        "Unknown var %s (previously %s)" % (new_name, old_name)
    return typed_ast.Var(new_name, type = tenv[new_name])

  def expr_Tuple():
    elts = annotate_children(expr.elts)
    elt_types = get_types(elts)
    t = tuple_type.make_tuple_type(elt_types)
    return typed_ast.Tuple(elts, type = t)

  def expr_Const():
    return typed_ast.Const(expr.value, type_conv.typeof(expr.value))

  def expr_Len():
    v = annotate_child(expr.value)
    t = v.type
    if t.__class__ is ArrayT:
      shape_t = make_tuple_type([Int64] * t.rank)
      shape = typed_ast.Attribute(v, 'shape', type = shape_t)
      return typed_ast.TupleProj(shape, 0, type = Int64)
    else:
      assert t.__class__ is TupleT, \
         "Unexpected argument type for 'len': %s" % t
      return typed_ast.Const(len(t.elt_types), type = Int64)

  def expr_Map():
    closure = annotate_child(expr.fn)
    new_args = annotate_args(expr.args, flat = True)
    axis = unwrap_constant(expr.axis)
    arg_types = get_types(new_args)
    result_type, typed_fn = specialize_Map(closure.type, arg_types)

    if axis is None and adverb_helpers.max_rank(arg_types) == 1:
      axis = 0
    return adverbs.Map(fn = make_typed_closure(closure, typed_fn),
                       args = new_args,
                       axis = axis,
                       type = result_type)

  def expr_Reduce():
    map_fn = annotate_child(expr.fn)
    combine_fn = annotate_child(expr.combine)
    new_args = annotate_args(expr.args, flat = True)
    arg_types = get_types(new_args)
    init = annotate_child(expr.init) if expr.init else None
    init_type = init.type if init else None
    result_type, typed_map_fn, typed_combine_fn = \
        specialize_Reduce(map_fn.type,
                          combine_fn.type,
                          arg_types, 
                          init_type)
    typed_map_closure = make_typed_closure (map_fn, typed_map_fn)
    typed_combine_closure = make_typed_closure(combine_fn, typed_combine_fn)
    axis = unwrap_constant(expr.axis)
    if axis is None and adverb_helpers.max_rank(arg_types) == 1:
      axis = 0
    if init_type and init_type != result_type and \
       array_type.rank(init_type) < array_type.rank(result_type):
      assert len(new_args) == 1
      assert axis == 0
      arg = new_args[0]
      first_elt = typed_ast.Index(arg, zero_i64, 
                                  type = arg.type.index_type(zero_i64))
      first_combine = specialize(combine_fn, (init_type, first_elt.type))
      first_combine_closure = make_typed_closure(combine_fn, first_combine)
      init = typed_ast.Call(first_combine_closure, (init, first_elt), 
                                 type = first_combine.return_type)
      slice_rest = typed_ast.Slice(start = one_i64, stop = none, step = one_i64, 
                                   type = array_type.SliceT(Int64, NoneType, Int64))
      rest = typed_ast.Index(arg, slice_rest, 
                             type = arg.type.index_type(slice_rest))
      new_args = (rest,)  
    return adverbs.Reduce(fn = typed_map_closure,
                          combine = typed_combine_closure,
                          args = new_args,
                          axis = axis,
                          type = result_type,
                          init = init)

  def expr_Scan():
    map_fn = annotate_child(expr.fn)
    combine_fn = annotate_child(expr.combine)
    emit_fn = annotate_child(expr.emit)
    new_args = annotate_args(expr.args, flat = True)
    arg_types = get_types(new_args)
    init = annotate_child(expr.init) if expr.init else None
    init_type = get_type(init) if init else None
    result_type, typed_map_fn, typed_combine_fn, typed_emit_fn = \
        specialize_Scan(map_fn.type, combine_fn.type, emit_fn.type,
                        arg_types, init_type)
    map_fn.fn = typed_map_fn
    combine_fn.fn = typed_combine_fn
    emit_fn.fn = typed_emit_fn
    axis = unwrap_constant(expr.axis)
    return adverbs.Scan(fn = make_typed_closure(map_fn, typed_map_fn),
                        combine = make_typed_closure(combine_fn,
                                                     typed_combine_fn),
                        emit = make_typed_closure(emit_fn, typed_emit_fn),
                        args = new_args,
                        axis = axis,
                        type = result_type,
                        init = init)

  def expr_AllPairs():
    closure = annotate_child(expr.fn)
    new_args = annotate_args (expr.args, flat = True)
    arg_types = get_types(new_args)
    assert len(arg_types) == 2
    xt,yt = arg_types
    result_type, typed_fn = specialize_AllPairs(closure.type, xt, yt)
    axis = unwrap_constant(expr.axis)
    return adverbs.AllPairs(make_typed_closure(closure, typed_fn),
                            args = new_args,
                            axis = axis,
                            type = result_type)

  result = dispatch(expr, prefix = "expr")
  assert result.type, "Missing type on %s" % result
  assert isinstance(result.type, Type), \
      "Unexpected type annotation on %s: %s" % (expr, result.type)
  return result

def annotate_stmt(stmt, tenv, var_map ):
  def infer_phi(result_var, val):
    """
    Don't actually rewrite the phi node, just add any necessary types to the
    type environment
    """

    new_val = annotate_expr(val, tenv, var_map)
    new_type = new_val.type
    old_type = tenv.get(result_var, Unknown)
    new_result_var = var_map.lookup(result_var)
    tenv[new_result_var]  = old_type.combine(new_type)

  def infer_phi_nodes(nodes, direction):
    for (var, values) in nodes.iteritems():
      infer_phi(var, direction(values))

  def infer_left_flow(nodes):
    return infer_phi_nodes(nodes, lambda (x, _): x)

  def infer_right_flow(nodes):
    return infer_phi_nodes(nodes, lambda (_, x): x)

  def annotate_phi_node(result_var, (left_val, right_val)):
    """
    Rewrite the phi node by rewriting the values from either branch, renaming
    the result variable, recording its new type, and returning the new name
    paired with the annotated branch values
    """

    new_left = annotate_expr(left_val, tenv, var_map)
    new_right = annotate_expr(right_val, tenv, var_map)
    old_type = tenv.get(result_var, Unknown)
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

  def annotate_lhs(lhs, rhs_type):
    lhs_class = lhs.__class__
    if lhs_class is untyped_ast.Tuple:
      if rhs_type.__class__ is TupleT:
        assert len(lhs.elts) == len(rhs_type.elt_types)
        new_elts = [annotate_lhs(elt, elt_type) for (elt, elt_type) in
                      zip(lhs.elts, rhs_type.elt_types)]
      else:
        assert rhs_type.__class__ is ArrayT, \
            "Unexpected right hand side type %s" % rhs_type
        elt_type = array_type.lower_rank(rhs_type, 1)
        new_elts = [annotate_lhs(elt, elt_type) for elt in lhs.elts]
      tuple_t = tuple_type.make_tuple_type(get_types(new_elts))
      return typed_ast.Tuple(new_elts, type = tuple_t)
    elif lhs_class is untyped_ast.Index:
      new_arr = annotate_expr(lhs.value, tenv, var_map)
      new_idx = annotate_expr(lhs.index, tenv, var_map)
      assert isinstance(new_arr.type, ArrayT), \
          "Expected array, got %s" % new_arr.type
      elt_t = new_arr.type.index_type(new_idx.type)
      return typed_ast.Index(new_arr, new_idx, type = elt_t)
    elif lhs_class is untyped_ast.Attribute:
      name = lhs.name
      struct = annotate_expr(lhs.value, tenv, var_map)
      struct_t = struct.type
      assert isinstance(struct_t, StructT), \
          "Can't access fields on value %s of type %s" % \
          (struct, struct_t)
      field_t = struct_t.field_type(name)
      return typed_ast.Attribute(struct, name, field_t)
    else:
      assert lhs_class is untyped_ast.Var, "Unexpected LHS: %s" % (lhs,)
      new_name = var_map.lookup(lhs.name)
      old_type = tenv.get(new_name, Unknown)
      new_type = old_type.combine(rhs_type)
      tenv[new_name] = new_type
      return typed_ast.Var(new_name, type = new_type)

  def stmt_Assign():
    rhs = annotate_expr(stmt.rhs, tenv, var_map)
    lhs = annotate_lhs(stmt.lhs, rhs.type)
    return typed_ast.Assign(lhs, rhs)

  def stmt_If():
    cond = annotate_expr(stmt.cond, tenv, var_map)
    assert isinstance(cond.type, ScalarT), \
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

  def stmt_ForLoop():
    infer_left_flow(stmt.merge)
    start = annotate_expr(stmt.start, tenv, var_map)
    stop = annotate_expr(stmt.stop, tenv, var_map)
    step = annotate_expr(stmt.step, tenv, var_map)
    lhs_t = start.type.combine(stop.type).combine(step.type)
    var = annotate_lhs(stmt.var, lhs_t)
    body = annotate_block(stmt.body, tenv, var_map)
    merge = annotate_phi_nodes(stmt.merge)

    return typed_ast.ForLoop(var, start, stop, step, body, merge)

  return dispatch(stmt, prefix="stmt")

def annotate_block(stmts, tenv, var_map):
  return [annotate_stmt(s, tenv, var_map) for s in stmts]

def infer_types(untyped_fn, types):
  """
  Given an untyped function and input types, propagate the types through the
  body, annotating the AST with type annotations.

  NOTE: The AST won't be in a correct state until a rewrite pass back-propagates
  inferred types throughout the program and inserts adverbs for scalar operators
  applied to arrays
  """

  var_map = VarMap()
  typed_args = untyped_fn.args.transform(rename_fn = var_map.rename)
  if untyped_fn.args.starargs:
    assert typed_args.starargs

  unbound_keywords = []
  def keyword_fn(local_name, value):
    unbound_keywords.append(local_name)
    return type_conv.typeof(value)

  tenv = typed_args.bind(types,
                         keyword_fn = keyword_fn,
                         starargs_fn = tuple_type.make_tuple_type)

  # keep track of the return
  tenv['$return'] = Unknown

  body = annotate_block(untyped_fn.body, tenv, var_map)
  arg_names = [local_name for local_name
               in
               typed_args.nonlocals + tuple(typed_args.positional)
               if local_name not in unbound_keywords]

  if len(unbound_keywords) > 0:
    default_assignments = []
    for local_name in unbound_keywords:
      t = tenv[local_name]
      python_value = typed_args.defaults[local_name]
      var = typed_ast.Var(local_name, type = t)
      typed_val = typed_ast.Const(python_value, type = t)
      stmt = typed_ast.Assign(var, typed_val)
      default_assignments.append(stmt)
    body = default_assignments + body

  input_types = tuple([tenv[arg_name] for arg_name in arg_names])

  # starargs are all passed individually and then packaged up
  # into a tuple on the first line of the function
  if typed_args.starargs:
    local_starargs_name = typed_args.starargs

    starargs_t = tenv[local_starargs_name]
    assert starargs_t.__class__ is TupleT, \
        "Unexpected starargs type %s" % starargs_t
    extra_arg_vars = []
    for (i, elt_t) in enumerate(starargs_t.elt_types):
      arg_name = "%s_elt%d" % (names.original(local_starargs_name), i)
      tenv[arg_name] = elt_t
      arg_var = typed_ast.Var(name = arg_name, type = elt_t)
      arg_names.append(arg_name)
      extra_arg_vars.append(arg_var)
    input_types = input_types + starargs_t.elt_types
    tuple_lhs = typed_ast.Var(name = local_starargs_name, type = starargs_t)
    tuple_rhs = typed_ast.Tuple(elts = extra_arg_vars, type = starargs_t)
    stmt = typed_ast.Assign(tuple_lhs, tuple_rhs)
    body = [stmt] + body

  return_type = tenv["$return"]
  # if nothing ever gets returned, then set the return type to None
  if isinstance(return_type,  UnknownT):
    body.append(typed_ast.Return(syntax_helpers.none))
    tenv["$return"] = NoneType
    return_type = NoneType

  return typed_ast.TypedFn(
    name = names.refresh(untyped_fn.name),
    body = body,
    arg_names = arg_names,
    input_types = input_types,
    return_type = return_type,
    type_env = tenv)

def _specialize(fn, arg_types):
  """
  Do the actual work of type specialization, whereas the wrapper 'specialize'
  pulls out untyped functions from closures, wraps argument lists in ActualArgs
  objects and performs memoization
  """

  if isinstance(fn, typed_ast.TypedFn):
    return fn
  typed_fundef = infer_types(fn, arg_types)
  from rewrite_typed import rewrite_typed
  coerced_fundef = rewrite_typed(typed_fundef)
  import simplify
  normalized = simplify.Simplify().apply(coerced_fundef)
  return normalized

def _get_fundef(fn):
  if isinstance(fn, (untyped_ast.Fn, typed_ast.TypedFn)):
    return fn
  else:
    assert isinstance(fn, str), \
        "Unexpected function %s : %s"  % (fn, fn.type)
    return untyped_ast.Fn.registry[fn]

def _get_closure_type(fn):
  if fn.__class__ is closure_type.ClosureT:
    return fn
  elif isinstance(fn, typed_ast.Closure):
    return fn.type
  elif isinstance(fn, typed_ast.Var):
    assert isinstance(fn.type, closure_type.ClosureT)
    return fn.type
  else:
    fundef = _get_fundef(fn)
    return closure_type.make_closure_type(fundef, [])

def specialize(fn, arg_types):
  if isinstance(fn, typed_ast.TypedFn):
    return fn
  if isinstance(arg_types, (list, tuple)):
    arg_types = ActualArgs(arg_types)
  closure_t = _get_closure_type(fn)
  if arg_types in closure_t.specializations:
    return closure_t.specializations[arg_types]

  full_arg_types = arg_types.prepend_positional(closure_t.arg_types)
  fundef = _get_fundef(closure_t.fn)
  typed =  _specialize(fundef, full_arg_types)
  closure_t.specializations[arg_types] = typed

  if config.print_specialized_function:
    print "=== Specialized %s for input types %s ===" % \
          (fundef.name, full_arg_types)
    print
    print repr(typed)
    print

  return typed

def infer_return_type(untyped, arg_types):
  """
  Given a function definition and some input types, gives back the return type
  and implicitly generates a specialized version of the function.
  """

  typed = specialize(untyped, arg_types)
  return typed.return_type

def specialize_Map(map_fn, array_types):
  elt_types = array_type.lower_ranks(array_types, 1)
  typed_map_fn = specialize(map_fn, elt_types)
  elt_result_t = typed_map_fn.return_type
  result_t = array_type.increase_rank(elt_result_t, 1)
  return result_t, typed_map_fn

def infer_Map(map_fn, array_types):
  t, _ = specialize_Map(map_fn, array_types)
  return t

def specialize_Reduce(map_fn, combine_fn, array_types, init_type = None):
  _, typed_map_fn = specialize_Map(map_fn, array_types)
  elt_type = typed_map_fn.return_type
  if init_type is None or isinstance(init_type, NoneT):
    acc_type = elt_type
  else:
    acc_type = elt_type.combine(init_type)

  typed_combine_fn = specialize(combine_fn, [acc_type, elt_type])
  new_acc_type = typed_combine_fn.return_type
  if new_acc_type != acc_type:
    typed_combine_fn = specialize(combine_fn, [new_acc_type, elt_type])
    new_acc_type = typed_combine_fn.return_type
  assert new_acc_type == acc_type
  return new_acc_type, typed_map_fn, typed_combine_fn

def infer_Reduce(map_fn, combine_fn, array_types, init_type = None):
  t, _, _ = specialize_Reduce(map_fn, combine_fn, array_types, init_type)
  return t

def specialize_Scan(map_fn, combine_fn, emit_fn, array_types, init_type = None):
  acc_type, typed_map_fn, typed_combine_fn = \
      specialize_Reduce(map_fn, combine_fn, array_types, init_type)
  typed_emit_fn = specialize(emit_fn, [acc_type])
  result_type = array_type.increase_rank(typed_emit_fn.return_type, 1)
  return result_type, typed_map_fn, typed_combine_fn, typed_emit_fn

def infer_Scan(map_fn, combine_fn, emit_fn, array_types, init_type = None):
  t, _, _, _ = specialize_Scan(map_fn, combine_fn, emit_fn,
                               array_types, init_type)
  return t

def specialize_AllPairs(fn, xtype, ytype):
  x_elt_t = array_type.lower_rank(xtype, 1)
  y_elt_t = array_type.lower_rank(ytype, 1)
  typed_map_fn = specialize(fn, [x_elt_t, y_elt_t])
  elt_result_t = typed_map_fn.return_type
  result_t = array_type.increase_rank(elt_result_t, 2)
  return result_t, typed_map_fn

def infer_AllPairs(fn, xtype, ytype):
  t, _ = specialize_AllPairs(fn, xtype, ytype)
  return t
