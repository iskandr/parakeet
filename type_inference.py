import adverbs
import adverb_helpers
import adverb_wrapper
import array_type
import closure_type
import core_types
import names
import prims
import syntax as untyped_ast
import syntax as typed_ast
import syntax_helpers
import tuple_type
import type_conv

from collections import OrderedDict
from common import dispatch
from function_registry import untyped_functions, find_specialization, \
                              add_specialization
from syntax_helpers import get_type, get_types, unwrap_constant
import function_registry
from args import ActualArgs

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

def linearize_arg_types(closure_t, args):
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
  if isinstance(closure_t, untyped_ast.Fn):
    untyped_fundef = closure_t
    closure_args = []
  elif isinstance(closure_t, str):
    untyped_fundef = function_registry.untyped_functions[closure_t]
    closure_args = []
  else:
    assert isinstance(closure_t, closure_type.ClosureT)
    if isinstance(closure_t.fn, str):
      untyped_fundef = function_registry.untyped_functions[closure_t.fn]
    else:
      untyped_fundef = closure_t.fn
    closure_args = closure_t.arg_types

  if isinstance(args, (list, tuple)):
    args = ActualArgs(args)

  if len(closure_args) > 0:
    args = args.prepend_positional(closure_args)

  def keyword_fn(_, v):
    return type_conv.typeof(v)

  linear_args, extra = untyped_fundef.args.linearize_values(args, keyword_fn = keyword_fn)
  return untyped_fundef, tuple(linear_args + extra)

def get_invoke_specialization(closure_t, arg_types):
  """
  for a given closure and the direct argument types it
  receives when invokes, return the specialization which
  will ultimately get called
  """
  untyped_fundef, full_arg_types = linearize_arg_types(closure_t, arg_types)
  return specialize(untyped_fundef, full_arg_types)

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

  if isinstance(fn, untyped_ast.Fn):
    fn = closure_type.ClosureT(fn.name, ())

  key = (fn, arg_types)
  if key in _invoke_type_cache:
    return _invoke_type_cache[key]

  if isinstance(fn, closure_type.ClosureT):
    closure_set = closure_type.ClosureSet(fn)
  else:
    assert isinstance(fn, closure_type.ClosureSet), \
      "Invoke expected closure, but got %s" % (fn,)
    closure_set = fn

  result_type = core_types.Unknown
  for closure_t in closure_set.closures:
    typed_fundef = get_invoke_specialization(closure_t, arg_types)
    result_type = result_type.combine(typed_fundef.return_type)
  _invoke_type_cache[key] = result_type
  return result_type

def annotate_expr(expr, tenv, var_map):

  def annotate_child(child_expr):
    return annotate_expr(child_expr, tenv, var_map)

  def annotate_children(child_exprs):
    return [annotate_expr(e, tenv, var_map) for e in child_exprs]

  def annotate_args(args):
    if isinstance(args, (list, tuple)):
      return map(annotate_child, args)
    else:
      return args.transform(annotate_child)

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
    t = closure_type.ClosureT(expr.fn, get_types(new_args))
    return typed_ast.Closure(expr.fn, new_args, type = t)

  def expr_Fn():
    t = closure_type.ClosureT(expr.name, [])
    return typed_ast.Closure(expr.name, [], type = t)

  def expr_Invoke():
    closure = annotate_child(expr.closure)
    args = annotate_args(expr.args)

    result_type = invoke_result_type(closure.type, get_types(args))

    # HERE YOU SHOULD FLATTEN THE ARGS! ... hopefully into the same
    # order as the underlying function?
    return typed_ast.Invoke(closure, args, type = result_type)

  def expr_Attribute():
    value = annotate_child(expr.value)
    assert isinstance(value.type, core_types.StructT)
    result_type = value.type.field_type(expr.name)
    return typed_ast.Attribute(value, expr.name, type = result_type)

  def expr_PrimCall():
    args = annotate_args(expr.args)
    arg_types = get_types(args)

    if all(isinstance(t, core_types.ScalarT) for t in arg_types):
      upcast_types = expr.prim.expected_input_types(arg_types)
      result_type = expr.prim.result_type(upcast_types)
      return typed_ast.PrimCall(expr.prim, args, type = result_type)
    else:
      assert all(not isinstance(t, core_types.NoneT) for t in arg_types)
      prim_fn = prims.prim_wrapper(expr.prim)

      max_rank = adverb_helpers.max_rank(arg_types)
      arg_names = adverb_wrapper.gen_data_arg_names(len(arg_types))
      untyped_broadcast_fn = \
        adverb_helpers.nested_maps(prim_fn, max_rank, arg_names)
      typed_broadcast_fn = specialize(untyped_broadcast_fn, arg_types)
      result_t = typed_broadcast_fn.return_type
      return typed_ast.Call(typed_broadcast_fn.name, args, type = result_t)

  def expr_Index():
    value = annotate_child(expr.value)
    index = annotate_child(expr.index)
    if isinstance(value.type, tuple_type.TupleT):
      assert isinstance(index.type, core_types.IntT)
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
    assert new_name in tenv, "Unknown var %s (previously %s)" % (new_name, old_name)
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
    new_args = annotate_args(expr.args)
    axis = unwrap_constant(expr.axis)
    arg_types = get_types(new_args)
    result_type = infer_map_type(closure.type, arg_types, axis)
    if axis is None and adverb_helpers.max_rank(arg_types) == 1:
      axis = 0
    return adverbs.Map(fn = closure,
                       args = new_args,
                       axis = axis,
                       type = result_type)

  def expr_Reduce():
    map_fn = annotate_child(expr.fn)
    combine_fn = annotate_child(expr.combine)
    new_args = annotate_args(expr.args)
    arg_types = get_types(new_args)
    axis = unwrap_constant(expr.axis)
    init = annotate_child(expr.init) if expr.init else None
    result_type = infer_reduce_type(
      map_fn.type,
      combine_fn.type,
      arg_types,
      axis,
      get_type(init) if init else None)
    if axis is None and adverb_helpers.max_rank(arg_types) == 1:
      axis = 0
    return adverbs.Reduce(fn = map_fn,
                          combine = combine_fn,
                          args = new_args,
                          axis = axis,
                          type = result_type,
                          init = init)

  def expr_Scan():
    map_fn = annotate_child(expr.fn)
    combine_fn = annotate_child(expr.combine)
    emit_fn = annotate_child(expr.emit)
    new_args = annotate_args(expr.args)
    arg_types = get_types(new_args)
    axis = unwrap_constant(expr.axis)
    init = annotate_child(expr.init) if expr.init else None
    result_type = infer_scan_type(map_fn, combine_fn, arg_types, axis)
    return adverbs.Scan(fn = map_fn,
                          combine = combine_fn,
                          emit = emit_fn,
                          args = new_args,
                          axis = axis,
                          type = result_type,
                          init = get_type(init) if init else None)

  def expr_AllPairs():
    closure = annotate_child(expr.fn)
    new_args = annotate_args (expr.args)
    arg_types = get_types(new_args)
    axis = unwrap_constant(expr.axis)
    result_type = infer_allpairs_type(closure.type, arg_types, axis)
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

        assert isinstance(new_arr.type, array_type.ArrayT), \
            "Expected array, got %s" % new_arr.type
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

def _infer_types(untyped_fn, types):
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
  tenv['$return'] = core_types.Unknown
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

  input_types = [tenv[arg_name] for arg_name in arg_names]

  # starargs are all passed individually and then packaged up
  # into a tuple on the first line of the function
  if typed_args.starargs:
    local_starargs_name = typed_args.starargs

    starargs_t = tenv[local_starargs_name]
    assert isinstance(starargs_t, tuple_type.TupleT), \
      "Unexpected starargs type %s" % starargs_t
    extra_arg_vars = []
    for (i, elt_t) in enumerate(starargs_t.elt_types):
      arg_name = "%s_elt%d" % (names.original(local_starargs_name), i)
      tenv[arg_name] = elt_t
      input_types.append(elt_t)
      arg_var = typed_ast.Var(name = arg_name, type = elt_t)
      arg_names.append(arg_name)
      extra_arg_vars.append(arg_var)
    tuple_lhs = typed_ast.Var(name = local_starargs_name, type = starargs_t)
    tuple_rhs = typed_ast.Tuple(elts = extra_arg_vars, type = starargs_t)
    stmt = typed_ast.Assign(tuple_lhs, tuple_rhs)
    body = [stmt] + body

  return_type = tenv["$return"]
  # if nothing ever gets returned, then set the return type to None
  if isinstance(return_type,  core_types.UnknownT):
    body.append(typed_ast.Return(syntax_helpers.none))
    tenv["$return"] = core_types.NoneType
    return_type = core_types.NoneType

  return typed_ast.TypedFn(
    name = names.refresh(untyped_fn.name),
    body = body,
    arg_names = arg_names,

    input_types = input_types,
    return_type = return_type,
    type_env = tenv)

def specialize(untyped, arg_types):
  if isinstance(untyped, str):
    untyped_id = untyped
    untyped = untyped_functions[untyped_id]
  else:
    assert isinstance(untyped, untyped_ast.Fn)
    untyped_id = untyped.name

  if isinstance(arg_types, (list, tuple)):
    arg_types = ActualArgs(arg_types)

  try:
    return find_specialization(untyped_id, arg_types)
  except:
    typed_fundef = _infer_types(untyped, arg_types)
    from rewrite_typed import rewrite_typed

    coerced_fundef = rewrite_typed(typed_fundef)

    import optimize
    # TODO: Also store the unoptimized version
    # so we can do adaptive recompilation
    opt = optimize.optimize(coerced_fundef, copy = False)
    add_specialization(untyped_id, arg_types, opt)

    # import lowering
    # lowered = lowering.lower(opt, copy = True, tile = False)
    # tiled = lowering.lower(opt, copy = True, tile = True)
    # opt.lowered = lowered
    # opt.tiled = tiled
    return opt

def infer_return_type(untyped, arg_types):
  """
  Given a function definition and some input types,
  gives back the return type
  and implicitly generates a specialized version of the
  function.
  """
  typed = specialize(untyped, arg_types)
  return typed.return_type

import adverb_semantics

class AdverbTypeSemantics(adverb_semantics.AdverbSemantics):
  none = core_types.NoneType

  def invoke(self, fn, arg_types):
    if isinstance(fn, typed_ast.TypedFn):
      input_types = fn.input_types
      assert all(in_t == arg_t for (in_t, arg_t) in zip(input_types,arg_types)), \
         "Expected types %s but got %s" % (input_types, arg_types)
      return fn.return_type
    if isinstance(fn, str):
      assert fn in function_registry.untyped_functions, \
      "Unknown function name %s" % (fn,)
      fn = function_registry.untyped_functions[fn]

    if isinstance(fn, untyped_ast.Fn):
      fn = closure_type.ClosureT(fn.name, ())
    return invoke_result_type(fn, arg_types)

  def size_along_axis(self, t, _):
    return core_types.Int64

  def int(self, _):
    return core_types.Int64

  def bool(self, _):
    return core_types.Bool

  def tuple(self, elt_types):
    return tuple_type.make_tuple_type(elt_types)

  def is_tuple(self, t):
    return isinstance(t, tuple_type.TupleT)

  def is_array(self, t):
    return isinstance(t, array_type.ArrayT)

  def is_none(self, t):
    return isinstance(t, core_types.NoneT)

  def shape(self, t):
    if self.is_array(t):
      return tuple_type.make_tuple_type([core_types.Int64] * t.rank)
    else:
      return tuple_type.make_tuple_type([])

  def elt_type(self, t):
    return t.elt_type if hasattr(t, 'elt_type') else t

  def alloc_array(self, elt_t, shape):
    return array_type.make_array_type(elt_t, len(shape))

  def setidx(self, arr, idx, val):
    pass

  def concat_tuples(self, t1, t2):
    return tuple_type.make_tuple_type(t1.elt_types + t2.elt_types)

  def slice_value(self, start_t, stop_t, step_t):
    return array_type.make_slice_type(start_t, stop_t, step_t)

  def index(self, arr, idx):
    return arr.index_type(idx)

  def check_equal_sizes(self, _):
    pass

  def rank(self, t):
    return t.rank if hasattr(t, 'rank') else 0

  def loop(self, start, _, loop_body):
    loop_body(start)

  class Accumulator:
    def __init__(self, v):
      self.value = v

    def get(self):
      return self.value

    def update(self, new_v):
      self.value = self.value.combine(new_v)

  def accumulate_loop(self, start, _, loop_body, init):
    acc = self.Accumulator(init)
    loop_body(acc, start)
    return acc.get()

adverb_type_semantics = AdverbTypeSemantics()

def infer_map_type(closure_t, arg_types, axis):
  untyped_fn, arg_types = linearize_arg_types(closure_t, arg_types)
  return adverb_type_semantics.eval_map(untyped_fn, arg_types, axis)

def infer_reduce_type(map_closure_t, combine_closure_t, arg_types, axis, init = None):
  map_fn, arg_types = linearize_arg_types(map_closure_t, arg_types)

  return adverb_type_semantics.eval_reduce(map_fn,
                                           combine_closure_t,
                                           init,
                                           arg_types,
                                           axis)


def infer_scan_type(
      map_closure_t,
      combine_closure_t,
      emit_closure_t,
      init_t,
      arg_types,
      axis):
  return adverb_type_semantics.eval_scan(map_closure_t,
                                         combine_closure_t,
                                         emit_closure_t,
                                         init_t,
                                         arg_types,
                                         axis)

def infer_allpairs_type(closure_t, arg_types, axis):
  if isinstance(arg_types, ActualArgs):
    arg_types = arg_types.positional
  assert len(arg_types) == 2
  [xtype, ytype] = arg_types
  axis = unwrap_constant(axis)
  return adverb_type_semantics.eval_allpairs(closure_t, xtype, ytype, axis)
