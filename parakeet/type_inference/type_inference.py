from .. import config, names,  prims, syntax

from ..builder import mk_prim_fn 
from ..ndtypes import (Type, 
                       array_type, closure_type, tuple_type, type_conv, 
                       Bool, IntT, Int64,  ScalarT, ArrayT,  
                       NoneType, NoneT, Unknown, UnknownT, 
                       TypeValueT, 
                       TupleT, make_tuple_type, 
                       lower_rank, make_array_type, 
                       ClosureT)

from ..syntax import adverb_helpers
from ..syntax import (UntypedFn, TypedFn, Closure,  Var, Map, Expr, 
                      ActualArgs, FormalArgs, MissingArgsError, TooManyArgsError)

from ..syntax.helpers import (get_type, get_types,  get_elt_types, 
                              one_i64, zero_i64, none, true, false, 
                              gen_data_arg_names, unwrap_constant)
from ..syntax.wrappers import build_untyped_prim_fn, build_untyped_cast_fn


from helpers import untyped_identity_function, make_typed_closure, _get_closure_type, _get_fundef
from linearize_args import linearize_actual_args, flatten_actual_args
from local_inference import LocalTypeInference 
from var_map import VarMap 


class TypeInferenceError(Exception):
  def __init__(self, msg, expr):
    self.msg = msg
    self.expr = expr 

_invoke_type_cache = {}
def invoke_result_type(fn, arg_types):
  if fn.__class__ is TypedFn:
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

  if fn.__class__ is ClosureT:
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

class TypeInference(LocalTypeInference):
  
  
  def transform_args(self, args, flat = False):
    if isinstance(args, (list, tuple)):
      return self.transform_expr_list(args)
    else:
      new_args = args.transform(self.transform_expr)
      if flat:
        return flatten_actual_args(new_args)
      else:
        return new_args

  def transform_keywords(self, kwds_dict):
    if kwds_dict is None:
      return {}
    else:
      result = {}
      for (k,v) in kwds_dict.iteritems():
        result[k] = self.transform_expr(v)
      return result

  
  def transform_fn(self, f):
  
    """
    If you're calling a Type, turn it into a wrapper function around the Cast expression. 
    If you're calling a Prim, turn it into a wrapper function around the PrimCall expression.
    """
    if isinstance(f, Type):
      expr = build_untyped_cast_fn(f)
    elif isinstance(f, prims.Prim):
      expr = build_untyped_prim_fn(f)
    else:
      expr = f 
    return self.transform_expr(expr)
  
  def transform_Call(self, expr):
    closure = self.transform_fn(expr.fn)
    args = self.transform_args(expr.args)
    if closure.type.__class__ is TypeValueT:
      assert args.__class__ is  ActualArgs
      assert len(args.positional) == 1
      assert len(args.keywords) == 0
      assert args.starargs is None 
      return self.cast(args.positional[0], closure.type.type)
    
    untyped_fn, args, arg_types = linearize_actual_args(closure, args)
    typed_fn = specialize(untyped_fn, arg_types)
    return syntax.Call(typed_fn, tuple(args), typed_fn.return_type)

  def transform_PrimCall(self, expr):
    args = self.transform_args(expr.args)

    arg_types = get_types(args)

    if all(isinstance(t, ScalarT) for t in arg_types):
      upcast_types = expr.prim.expected_input_types(arg_types)
      result_type = expr.prim.result_type(upcast_types)
      return syntax.PrimCall(expr.prim, args, type = result_type)
    elif expr.prim == prims.is_:
      if arg_types[0] != arg_types[1]:
        return false
      elif arg_types[0] == NoneType:
        return true
      else:
        return syntax.PrimCall(prims.is_, args, type = Bool)
    elif all(t.rank == 0 for t in arg_types):
      # arguments should then be tuples
      assert len(arg_types) == 2, \
        "Expected two arguments but got [%s] in %s" % (", ".join(str(t) for t in arg_types), expr)
      xt, yt = arg_types
      x, y = args
      assert xt.__class__ is TupleT, \
        "Unexpected argument types (%s,%s) for operator %s" % (xt, yt, expr.prim)
      assert yt.__class__ is TupleT, \
        "Unexepcted argument types (%s,%s) for operator %s" % (xt, yt, expr.prim)
      x_elts = self.tuple_elts(x)
      y_elts = self.tuple_elts(y)
      nx = len(x_elts)
      ny = len(y_elts)
      if expr.prim is prims.add:
        return self.tuple( tuple(x_elts) + tuple(y_elts) )
      elif expr.prim is prims.equal:
        assert len(x_elts) == len(y_elts), \
          "Can't compare tuple of unequal lengths %d and %d" % (nx, ny)
        result = true  
        for (xi, yi) in zip(x_elts, y_elts):
          elts_eq = syntax.PrimCall(prims.equal, (xi, yi), type=Bool)
          result = syntax.PrimCall(prims.logical_and, (result, elts_eq), type=Bool) 
        return result  
      elif expr.prim is prims.not_equal:
        assert len(x_elts) == len(y_elts), \
          "Can't compare tuple of unequal lengths %d and %d" % (nx, ny)
        result = false  
        for (xi, yi) in zip(x_elts, y_elts):
          elts_eq = syntax.PrimCall(prims.not_equal, (xi, yi), type=Bool)
          result = syntax.PrimCall(prims.logical_or, (result, elts_eq), type=Bool) 
        return result  
      
      else:
        assert False, "Unsupported tuple operation %s" % expr  
    else:
      assert all(t.__class__ is not NoneT for t in arg_types), \
        "Invalid argument types for prim %s: %s" % (expr.prim, arg_types,)
      elt_types = [t.elt_type if isinstance(t, ArrayT) else t for t in arg_types]
      typed_prim_fn = mk_prim_fn(expr.prim, elt_types)
      max_rank = adverb_helpers.max_rank(arg_types)
      if max_rank == 0:
        return self.call(typed_prim_fn, args)
      elif max_rank == 1:
        axis = zero_i64 
      else:
        axis = none 
      result_t = make_array_type(typed_prim_fn.return_type, max_rank)
      return Map(fn = typed_prim_fn, args = args, type = result_t, axis = axis)
      

  def transform_Zip(self, expr):
    assert isinstance(expr.values, (list,tuple)), "Expected multiple values but got %s" % expr.values
    typed_values = self.transform_expr_tuple(expr.values)
    # if you're just zipping tuples together, 
    # then the result can also be a tuple
    if all(isinstance(v.type, TupleT) for v in typed_values):
      # keep the shortest tuple
      n = min(len(v.type.elt_types) for v in typed_values)
      zip_inputs = []
      for v in typed_values:
        zip_inputs.append(self.tuple_elts(v)[:n])
      return self.tuple([self.tuple(group) for group in zip(*zip_inputs)])
    
    # if any are tuples, that puts a max length on any sequence
    elif any(isinstance(v.type, TupleT) for v in typed_values):
      # keep the shortest tuple
      n = min(len(v.type.elt_types) for v in typed_values if isinstance(v.type, TupleT))
      assert False, "Zipping of mixed tuples and arrays not yet supported"  
    else:
      assert all(isinstance(v.type, ArrayT) for v in typed_values), \
        "Expected all inputs to zip to be arrays but got %s" % \
        ", ".join(str(v.type) for v in typed_values)
      
      elt_t = make_tuple_type([v.type.elt_type for v in typed_values])  
      result_t = make_array_type(elt_t, 1)
      def tupler(*args):
        return args 
      from ..frontend import ast_conversion
      untyped = ast_conversion.translate_function_value(tupler)
      typed = specialize(untyped, [v.type.elt_type for v in typed_values])
      result = Map(fn = typed, args = typed_values, axis = zero_i64, type = result_t)
      assert False, "Materializing the result of zipping arrays not yet supported"
      #return result
 
  def transform_IndexMap(self, expr):
    shape = self.transform_expr(expr.shape)
    if not isinstance(shape.type, TupleT):
      assert isinstance(shape.type, ScalarT), "Invalid shape for IndexMap: %s : %s" % (shape, shape.type)
      shape = self.tuple((shape,))
    closure = self.transform_fn(expr.fn)
    shape_t = shape.type
    if isinstance(shape_t, IntT):
      shape = self.cast(shape, Int64)
      n_indices = 1
    else:
      assert isinstance(shape_t, TupleT), "Expected shape to be tuple, instead got %s" % (shape_t,)
      assert all(isinstance(t, ScalarT) for t in shape_t.elt_types)
      n_indices = len(shape_t.elt_types)
      if not all(t == Int64 for t in shape_t.elt_types):
        elts = tuple(self.cast(elt, Int64) for elt in self.tuple_elts(shape))
        shape = self.tuple(elts)
    result_type, typed_fn = specialize_IndexMap(closure.type, n_indices)
    return syntax.IndexMap(shape = shape, 
                           fn = make_typed_closure(closure, typed_fn), 
                           type = result_type)
  
  def transform_IndexReduce(self, expr):
    shape = self.transform_expr(expr.shape)
    map_fn_closure = self.transform_fn(expr.fn if expr.fn else untyped_identity_function)
    combine_closure = self.transform_fn(expr.combine)
    init = self.transform_if_expr(expr.init)
    shape_t = shape.type
    if isinstance(shape_t, IntT):
      shape = self.cast(shape, Int64)
      n_indices = 1
    else:
      assert isinstance(shape_t, TupleT)
      assert all(isinstance(t, ScalarT) for t in shape_t.elt_types)
      n_indices = len(shape_t.elt_types)
      if not all(t == Int64 for t in shape_t.elt_types):
        elts = tuple(self.cast(elt, Int64) for elt in self.tuple_elts(shape))
        shape = self.tuple(elts)
    result_type, typed_fn, typed_combine = \
      specialize_IndexReduce(map_fn_closure.type, combine_closure, n_indices, init)
    if not self.is_none(init):
      init = self.cast(init, result_type)
    return syntax.IndexReduce(shape = shape, 
                              fn = make_typed_closure(map_fn_closure, typed_fn),
                              combine = make_typed_closure(combine_closure, typed_combine),
                              init = init,  
                              type = result_type)
  
  def transform_Map(self, expr):
    closure = self.transform_fn(expr.fn)
    new_args = self.transform_args(expr.args, flat = True)
    arg_types = get_types(new_args)
    assert len(arg_types) > 0, "Map requires array arguments"
    # if all arguments are scalars just handle map as a regular function call
    if all(isinstance(t, ScalarT) for t in arg_types):
      return self.invoke(closure, new_args)
    # if any arguments are tuples then all of them should be tuples of same len
    elif any(isinstance(t, TupleT) for t in arg_types):
      assert all(isinstance(t, TupleT) for t in arg_types), \
        "Map doesn't support input types %s" % (arg_types,)
      nelts = len(arg_types[0].elt_types)
      assert all(len(t.elt_types) == nelts for t in arg_types[1:]), \
       "Tuple arguments to Map must be of same length"
      zipped_elts = []
      for i in xrange(nelts):
        zipped_elts.append([self.tuple_proj(arg,i) for arg in new_args])
      return self.tuple([self.invoke(closure, elts) for elts in zipped_elts])
    axis = self.transform_if_expr(expr.axis)
    axes = self.normalize_axes(new_args, axis)
    result_type, typed_fn = specialize_Map(closure.type, arg_types, axes)
    return syntax.Map(fn = make_typed_closure(closure, typed_fn),
                       args = new_args,
                       axis = axis,
                       type = result_type)


  
    
  def transform_Reduce(self, expr):
    assert len(expr.args) > 0, "Can't have Reduce without any arguments %s" % expr 
    new_args = self.transform_args(expr.args, flat = True)
    arg_types = get_types(new_args)
    axis = self.transform_if_expr(expr.axis)

    map_fn = self.transform_fn(expr.fn if expr.fn else untyped_identity_function) 
    
    
    # if there aren't any arrays, just treat this as a function call
    if all(isinstance(t, ScalarT) for t in arg_types):
      return self.invoke(map_fn, new_args)
    
    combine_fn = self.transform_fn(expr.combine)
    if self.is_none(expr.init):
      init = none 
    else: 
      init = self.transform_expr(expr.init)
      
    axes = self.normalize_axes(new_args, axis)
    result_type, typed_map_fn, typed_combine_fn = \
      specialize_Reduce(map_fn.type,
                        combine_fn.type,
                        arg_types, 
                        axes, 
                        init.type)  

    typed_map_closure = make_typed_closure (map_fn, typed_map_fn)
    typed_combine_closure = make_typed_closure(combine_fn, typed_combine_fn)
    
    # if we encounter init = 0 for a Reduce which produces an array
    # then need to broadcast to get an initial value of the appropriate rank 
    if init.type.__class__ is not NoneT and \
       init.type != result_type and \
       array_type.rank(init.type) < array_type.rank(result_type):
      assert len(new_args) == 1, "Can't have more than one arg in " % expr  
      arg = new_args[0]
      first_elt = self.slice_along_axis(arg, axis, zero_i64)
      first_combine = specialize(combine_fn, (init.type, first_elt.type))
      
      first_combine_closure = make_typed_closure(combine_fn, first_combine)
      init = self.call(first_combine_closure, (init, first_elt))
      slice_rest = syntax.Slice(start = one_i64, stop = none, step = one_i64, 
                                   type = array_type.SliceT(Int64, NoneType, Int64))
      rest = self.slice_along_axis(arg, axis, slice_rest)
      new_args = (rest,)  
    
    return syntax.Reduce(fn = typed_map_closure,
                         combine = typed_combine_closure,
                         args = new_args,
                         axis = axis,
                         type = result_type,
                         init = init)

  def transform_Scan(self, expr):
    map_fn = self.transform_fn(expr.fn if expr.fn else untyped_identity_function)
    combine_fn = self.transform_fn(expr.combine)
    emit_fn = self.transform_fn(expr.emit)
    new_args = self.transform_args(expr.args, flat = True)
    arg_types = get_types(new_args)
    
    init = self.transform_expr(expr.init) if expr.init else None
    
    init_type = get_type(init) if init else None
    
    axis = self.transform_if_expr(expr.axis)
    axes = self.normalize_axes(new_args, axis)
    result_type, typed_map_fn, typed_combine_fn, typed_emit_fn = \
        specialize_Scan(map_fn.type, combine_fn.type, emit_fn.type,
                        arg_types, axes, init_type)
    map_fn.fn = typed_map_fn
    combine_fn.fn = typed_combine_fn
    emit_fn.fn = typed_emit_fn
    return syntax.Scan(fn = make_typed_closure(map_fn, typed_map_fn),
                       combine = make_typed_closure(combine_fn,
                                                    typed_combine_fn),
                       emit = make_typed_closure(emit_fn, typed_emit_fn),
                       args = new_args,
                       axis = axis,
                       type = result_type,
                       init = init)

  def transform_OuterMap(self, expr):
    closure = self.transform_fn(expr.fn)
    new_args = self.transform_args (expr.args, flat = True)
    arg_types = get_types(new_args)
    n_args = len(arg_types)
    assert n_args > 0
    axis = self.transform_if_expr(expr.axis)
    axes = self.normalize_axes(new_args, axis)
    result_type, typed_fn = specialize_OuterMap(closure.type, arg_types, axes)
    result = syntax.OuterMap(fn = make_typed_closure(closure, typed_fn),
                             args = new_args,
                             axis = axis,
                             type = result_type)
    return result 

  
  


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
    assert typed_args.starargs, "Missing star-args in call to %s(%s)" % (untyped_fn.name, typed_args,)

  unbound_keywords = []
  def keyword_fn(local_name, value):
    unbound_keywords.append(local_name)

    if isinstance(value, Expr):
      value = unwrap_constant(value) 
    return type_conv.typeof(value)

  try: 
    tenv = typed_args.bind(types,
                           keyword_fn = keyword_fn,
                           starargs_fn = tuple_type.make_tuple_type)
  except (MissingArgsError, TooManyArgsError) as e:
    e.fn_name = untyped_fn.name 
    raise e
  except: 
    print "Error while calling %s with types %s" % (untyped_fn, types)
    raise
  # keep track of the return
  tenv['$return'] = Unknown
  annotator = TypeInference(tenv, var_map)
  body = annotator.transform_block(untyped_fn.body)
  arg_names = [local_name for local_name
               in
               typed_args.nonlocals + tuple(typed_args.positional)
               if local_name not in unbound_keywords]
  if len(unbound_keywords) > 0:
    default_assignments = []
    from ..frontend.ast_conversion import value_to_syntax 
    for local_name in unbound_keywords:
      t = tenv[local_name]
      python_value = typed_args.defaults[local_name]
      var = Var(local_name, type = t)
      parakeet_value = value_to_syntax(python_value)
      parakeet_value.type = t 
      stmt = syntax.Assign(var, parakeet_value)
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
      arg_var = Var(name = arg_name, type = elt_t)
      arg_names.append(arg_name)
      extra_arg_vars.append(arg_var)
    input_types = input_types + starargs_t.elt_types
    tuple_lhs = Var(name = local_starargs_name, type = starargs_t)
    tuple_rhs = syntax.Tuple(elts = extra_arg_vars, type = starargs_t)
    stmt = syntax.Assign(tuple_lhs, tuple_rhs)
    body = [stmt] + body

  return_type = tenv["$return"]
  # if nothing ever gets returned, then set the return type to None
  if return_type.__class__ is  UnknownT:
    body.append(syntax.Return(none))
    tenv["$return"] = NoneType
    return_type = NoneType

  return TypedFn(
    name = names.refresh(untyped_fn.name),
    body = body,
    arg_names = arg_names,
    input_types = input_types,
    return_type = return_type,
    type_env = tenv)

def _specialize(fn, arg_types, return_type = None):
  """
  Do the actual work of type specialization, whereas the wrapper 'specialize'
  pulls out untyped functions from closures, wraps argument lists in ActualArgs
  objects and performs memoization
  """
  if fn.__class__ is TypedFn:
    return fn
  typed_fundef = infer_types(fn, arg_types)
  from rewrite_typed import rewrite_typed
  return rewrite_typed(typed_fundef, return_type)




def specialize(fn, arg_types, return_type = None):

  if config.print_before_specialization:
    if return_type:
      print "==== Specializing", fn, "for input types", arg_types, "and return type", return_type
    else:  
      print "=== Specializing", fn, "for types", arg_types 
  if fn.__class__ is TypedFn:
    assert len(fn.input_types) == len(arg_types)
    assert all(t1 == t2 for t1,t2 in zip(fn.input_types, arg_types))
    if return_type is not None:
      assert fn.return_type == return_type 
    return fn
  
  if isinstance(arg_types, (list, tuple)):
    arg_types = ActualArgs(arg_types)
  closure_t = _get_closure_type(fn)
  key = arg_types, return_type
  if key in closure_t.specializations:
    return closure_t.specializations[key]

  full_arg_types = arg_types.prepend_positional(closure_t.arg_types)
  fundef = _get_fundef(closure_t.fn)
  typed =  _specialize(fundef, full_arg_types, return_type)
  closure_t.specializations[key] = typed

  if config.print_specialized_function:
    if return_type:
      print "=== Specialized %s for input types %s and return type %s ==="  % \
          (fundef.name, full_arg_types, return_type)
    else:
      print "=== Specialized %s for input types %s ==="  % \
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
  return specialize(untyped, arg_types).return_type

def specialize_IndexMap(fn, n_indices):
  idx_type = make_tuple_type( (Int64,) * n_indices) if n_indices > 1 else Int64
  typed_fn = specialize(fn, (idx_type,))
  result_type = array_type.increase_rank(typed_fn.return_type, n_indices)
  return result_type, typed_fn

def specialize_IndexReduce(fn, combine, n_indices, init = None):
  idx_type = make_tuple_type( (Int64,) * n_indices) if n_indices > 1 else Int64
  if init is None or init.type.__class__ is  NoneT:
    typed_fn = specialize(fn, (idx_type,))
  else:
    typed_fn = specialize(fn, (idx_type,), return_type = init.type)
  elt_type = typed_fn.return_type
  typed_combine = specialize(combine, (elt_type, elt_type))
  return elt_type, typed_fn, typed_combine 

def peel_adverb_input_types(array_types, axes):
  assert len(array_types) == len(axes), \
    "Mismatch between types %s and axes %s" % (array_types, axes)
  result = []
  for t, axis in zip(array_types, axes):
    if axis is None:
      result.append(array_type.elt_type(t))
    else:
      result.append(array_type.lower_rank(t,1))
  return tuple(result)

def rank(t):
  if t.__class__ is ArrayT:
    return t.rank 
  else:
    return 0 

def increase_adverb_output_rank(array_types, axes, elt_result_type, cartesian_product = False):
  delta_rank = 0
  assert len(array_types) == len(axes), \
    "Mismatch between types %s and axes %s" % (array_types, axes)
  for (t,axis) in zip(array_types, axes):
    if axis is None: r = rank(t)
    else: r = 1
    if cartesian_product: delta_rank += r 
    else: delta_rank = max(r, delta_rank)
  return array_type.increase_rank(elt_result_type, delta_rank)

def specialize_Map(map_fn, array_types, axes):
  
  elt_types = peel_adverb_input_types(array_types, axes)
  typed_map_fn = specialize(map_fn, elt_types)
  elt_result_t = typed_map_fn.return_type
  result_t = increase_adverb_output_rank(array_types, axes, elt_result_t)
  return result_t, typed_map_fn

def specialize_Reduce(map_fn, combine_fn, array_types, axes, init_type = None):
  _, typed_map_fn = specialize_Map(map_fn, array_types, axes)
  elt_type = typed_map_fn.return_type
  if init_type is None or init_type.__class__ is NoneT:
    acc_type = elt_type
  else:
    acc_type = init_type
  typed_combine_fn = specialize(combine_fn, [acc_type, elt_type])
  new_acc_type = typed_combine_fn.return_type
  if new_acc_type != acc_type:
    typed_combine_fn = specialize(combine_fn, [new_acc_type, elt_type])
    new_acc_type = typed_combine_fn.return_type
  return new_acc_type, typed_map_fn, typed_combine_fn

def specialize_Scan(map_fn, combine_fn, emit_fn, array_types, axes,  init_type = None):
  acc_type, typed_map_fn, typed_combine_fn = \
      specialize_Reduce(map_fn, combine_fn, array_types, axes, init_type)
  typed_emit_fn = specialize(emit_fn, [acc_type])
  elt_result_t = typed_emit_fn.return_type
  result_t = increase_adverb_output_rank(array_types, axes, elt_result_t)
  return result_t, typed_map_fn, typed_combine_fn, typed_emit_fn

def specialize_OuterMap(fn, array_types, axes):
  elt_types = peel_adverb_input_types(array_types, axes)
  typed_map_fn = specialize(fn, elt_types)
  elt_result_t = typed_map_fn.return_type
  result_t = increase_adverb_output_rank(array_types, axes, elt_result_t, cartesian_product = True)
  return result_t, typed_map_fn

