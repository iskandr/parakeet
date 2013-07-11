import adverb_helpers
import array_type
import ast_conversion
import closure_type
import config
import core_types 
import names
import prims
import syntax
import syntax_helpers
import tuple_type
import type_conv

from args import ActualArgs, FormalArgs, MissingArgsError
from array_type import ArrayT
from closure_type import ClosureT 
from core_types import Type, Bool, IntT, Int64,  ScalarT
from core_types import NoneType, NoneT, Unknown, UnknownT
from core_types import combine_type_list, StructT
from stride_specialization import specialize
from syntax import Fn, TypedFn, Closure, ClosureElt, Var 
from syntax_helpers import get_type, get_types, unwrap_constant
from syntax_helpers import one_i64, zero_i64, none, true, false, is_false, is_true, is_zero
from transform import Transform
from tuple_type import TupleT, make_tuple_type

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
  elif closure.__class__ is Closure:
    fn = closure.fn 
    closure_args = closure.args 
  elif closure.type.__class__ is closure_type.ClosureT:
    fn, arg_types = closure.type.fn, closure.type.arg_types
    closure_args = \
        [ClosureElt(closure, i, type = arg_t)
         for (i, arg_t) in enumerate(arg_types)]
  else:
    fn = closure
    closure_args = []
    # fn = syntax.Fn.registry[fn]
  return fn, closure_args

def make_typed_closure(untyped_closure, typed_fn):
  if untyped_closure.__class__ is syntax.Fn:
    return typed_fn

  assert isinstance(untyped_closure, syntax.Expr) and \
      isinstance(untyped_closure.type, closure_type.ClosureT)
  _, closure_args = unpack_closure(untyped_closure)
  if len(closure_args) == 0:
    return typed_fn
  else:
    t = closure_type.make_closure_type(typed_fn, get_types(closure_args))
    return Closure(typed_fn, closure_args, t)

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
  return [syntax.TupleProj(tup, i, t)
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
  if isinstance(fn, syntax.TypedFn):
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

def identity(x):
  return x
untyped_identity_function = ast_conversion.translate_function_value(identity)


class Annotator(Transform):
  
  def __init__(self, tenv, var_map):
    Transform.__init__(self)
    self.type_env = tenv 
    self.var_map = var_map
    
  
  def transform_expr(self, expr):
    if not isinstance(expr, syntax.Expr):
      expr = ast_conversion.value_to_syntax(expr)
    
    result = Transform.transform_expr(self, expr)  
    assert result.type is not None,  \
      "Unsupported expression encountered during type inference: %s" % (expr,)
    return result 
  
  
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

  def keyword_types(self, kwds):
    keyword_types = {}
    for (k,v) in kwds.iteritems():
      keyword_types[k] = get_type(v)
    return keyword_types
  
  def transform_DelayUntilTyped(self, expr):
    new_values = self.transform_expr_tuple(expr.values)
    new_syntax = expr.fn(*new_values)
    assert new_syntax.type is not None
    return new_syntax
  
  def transform_TypeValue(self, expr):
    t = expr.type_value 
    assert isinstance(t, core_types.Type), "Invalid type value %s" % (t,)
    return syntax.TypeValue(t, type=core_types.TypeValueT(t))
    
  def transform_Closure(self, expr):
    new_args = self.transform_expr_list(expr.args)
    t = closure_type.make_closure_type(expr.fn, get_types(new_args))
    return Closure(expr.fn, new_args, type = t)

  def transform_Arith(self, expr):
    untyped_fn = prims.prim_wrapper(expr)
    t = closure_type.make_closure_type(untyped_fn, ())
    return Closure(untyped_fn, (), type = t)

  def transform_Fn(self, expr):
    return expr 
    #t = closure_type.make_closure_type(expr, ())
    #return Closure(expr, [], type = t)

  def transform_Call(self, expr):
    closure = self.transform_expr(expr.fn)
    args = self.transform_args(expr.args)
    if closure.type.__class__ is core_types.TypeValueT:
      assert isinstance(args, ActualArgs)
      assert len(args.positional) == 1
      assert len(args.keywords) == 0
      assert args.starargs is None 
      return self.cast(args.positional[0], closure.type.type)
    
    untyped_fn, args, arg_types = linearize_actual_args(closure, args)

    typed_fn = specialize(untyped_fn, arg_types)
    return syntax.Call(typed_fn, args, typed_fn.return_type)

  def transform_Attribute(self, expr):
    value = self.transform_expr(expr.value)
    assert isinstance(value.type, StructT)
    result_type = value.type.field_type(expr.name)
    return syntax.Attribute(value, expr.name, type = result_type)
  
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
      assert len(arg_types) == 2
      xt, yt = arg_types
      x, y = args 
      assert isinstance(xt, TupleT), \
        "Unexpected argument types (%s,%s) for operator %s" % (xt, yt, expr.prim)
      assert isinstance(yt, TupleT), \
        "Unexepcted argument types (%s,%s) for operator %s" % (xt, yt, expr.prim)
      x1 = syntax.TupleProj(x, 0, type=xt.elt_types[0])
      x2 = syntax.TupleProj(x, 1, type=xt.elt_types[1])
      y1 = syntax.TupleProj(y, 0, type=yt.elt_types[0])
      y2 = syntax.TupleProj(y, 1, type=yt.elt_types[1]) 
      if expr.prim == prims.equal:
        first = syntax.PrimCall(prims.equal, (x1, y1), type=Bool)
        second = syntax.PrimCall(prims.equal, (x2, y2), type=Bool)
        return syntax.PrimCall(prims.logical_and, (first, second), type=Bool)
      else:
        assert False, "Unsupport tuple operation %s" % expr  
    else:
      assert all(not isinstance(t, NoneT) for t in arg_types), \
        "Invalid argument types for prim %s: %s" % (expr.prim, arg_types,)
      prim_fn = prims.prim_wrapper(expr.prim)

      max_rank = adverb_helpers.max_rank(arg_types)
      arg_names = syntax_helpers.gen_data_arg_names(len(arg_types))
      untyped_broadcast_fn = \
          adverb_helpers.nested_maps(prim_fn, max_rank, arg_names)
      typed_broadcast_fn = specialize(untyped_broadcast_fn, arg_types)
      result_t = typed_broadcast_fn.return_type
      return syntax.Call(typed_broadcast_fn, args, type = result_t)

  def transform_Index(self, expr):
    value = self.transform_expr(expr.value)
    index = self.transform_expr(expr.index)
    if isinstance(value.type, TupleT):
      assert isinstance(index.type, IntT)
      assert isinstance(index, syntax.Const)
      i = index.value
      assert isinstance(i, int)
      elt_types = value.type.elt_types
      assert i < len(elt_types), \
          "Can't get element %d of length %d tuple %s : %s" % \
          (i, len(elt_types), value, value.type)
      elt_t = value.type.elt_types[i]
      return syntax.TupleProj(value, i, type = elt_t)
    else:
      result_type = value.type.index_type(index.type)
      return syntax.Index(value, index, type = result_type)

  def transform_Array(self, expr):
    new_elts = self.transform_args(expr.elts)
    elt_types = get_types(new_elts)
    common_t = combine_type_list(elt_types)
    array_t = array_type.increase_rank(common_t, 1)
    return syntax.Array(new_elts, type = array_t)

  def transform_AllocArray(self, expr):
    elt_type = expr.elt_type
    assert isinstance(elt_type, core_types.ScalarT), \
      "Invalid array element type  %s" % (elt_type)
      
    shape = self.transform_expr(expr.shape)
    if isinstance(shape, core_types.ScalarT):
      shape = self.cast(shape, Int64)
      shape = self.tuple((shape,), "array_shape")
    assert isinstance(shape, TupleT), \
      "Invalid shape %s" % (shape,)
    rank = len(shape.elt_types)
    t = array_type.make_array_type(elt_type, rank)
    return syntax.AllocArray(shape, elt_type, type = t)
  
  def transform_Range(self, expr):
    start = self.transform_expr(expr.start) if expr.start else None
    stop = self.transform_expr(expr.stop) if expr.stop else None
    step = self.transform_expr(expr.step) if expr.step else None
    array_t = array_type.ArrayT(Int64, 1)
    return syntax.Range(start, stop, step, type = array_t)

  def transform_Slice(self, expr):
    start = self.transform_expr(expr.start)
    stop = self.transform_expr(expr.stop)
    step = self.transform_expr(expr.step)
    slice_t = array_type.make_slice_type(start.type, stop.type, step.type)
    return syntax.Slice(start, stop, step, type = slice_t)

  def transform_Var(self, expr):
    old_name = expr.name
    if old_name not in self.var_map._vars:
      raise names.NameNotFound(old_name)
    new_name = self.var_map.lookup(old_name)
    assert new_name in self.type_env, \
        "Unknown var %s (previously %s)" % (new_name, old_name)
    t = self.type_env[new_name]
    return Var(new_name, type = t)

  def transform_Tuple(self, expr):
    elts = self.transform_expr_list(expr.elts)
    return self.tuple(elts)
    #elt_types = get_types(elts)
    #t = tuple_type.make_tuple_type(elt_types)
    #return syntax.Tuple(elts, type = t)

  def transform_Const(self, expr):
    return syntax.Const(expr.value, type_conv.typeof(expr.value))
  
  def transform_ConstArrayLike(self, expr):
    typed_array_value = self.transform_expr(expr.array)
    t = typed_array_value.type
    return syntax.ConstArrayLike(typed_array_value, expr.value, type = t)

  def transform_Reshape(self, expr):
    array = self.transform_expr(expr.array)
    shape = self.transform_expr(expr.shape)
    rank = len(shape.type.elt_types)
    assert isinstance(array.type, array_type.ArrayT)
    t = array_type.make_array_type(array.elt_type, rank)
    return syntax.Reshape(array, shape, type = t)
  
  def transform_Ravel(self, expr):
    array = self.transform_expr(expr.array)
    if not isinstance(array.type, array_type.ArrayT):
      print "Warning: Can't ravel/flatten an object of type %s" % array.type 
      return array 
    t = array_type.make_array_type(array.type.elt_type, 1)
    return syntax.Ravel(array, type = t)
  
  
  def transform_Cast(self, expr):
    v = self.transform_expr(expr.value)
    return syntax.Cast(v, type = expr.type)
  
  def transform_Len(self, expr):
    v = self.transform_expr(expr.value)
    t = v.type
    if t.__class__ is ArrayT:
      shape_t = make_tuple_type([Int64] * t.rank)
      shape = syntax.Attribute(v, 'shape', type = shape_t)
      return syntax.TupleProj(shape, 0, type = Int64)
    else:
      assert t.__class__ is TupleT, \
         "Unexpected argument type for 'len': %s" % t
      return syntax.Const(len(t.elt_types), type = Int64)
 
  def transform_IndexMap(self, expr):
    shape = self.transform_expr(expr.shape)
    if not isinstance(shape.type, TupleT):
      assert isinstance(shape.type, ScalarT), "Invalid shape for IndexMap: %s" % (shape,)
      shape = self.tuple((shape,))
    closure = self.transform_expr(expr.fn)
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
    map_fn_closure = self.transform_expr(expr.fn)
    combine_closure = self.transform_expr(expr.combine)
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
    closure = self.transform_expr(expr.fn)
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
    result_type, typed_fn = specialize_Map(closure.type, arg_types)
    if axis is None or self.is_none(axis):
      assert adverb_helpers.max_rank(arg_types) == 1
      axis = syntax_helpers.zero_i64
    return syntax.Map(fn = make_typed_closure(closure, typed_fn),
                       args = new_args,
                       axis = axis,
                       type = result_type)


  def flatten_Reduce(self, map_fn, combine, x, init):
    """Turn an axis-less reduction into a IndexReduce"""
    shape = self.shape(x)
    n_indices = self.rank(x)
    # build a function from indices which picks out the data elements
    # need for the original map_fn
 
    
    outer_closure_args = self.closure_elts(map_fn)
    args_obj = FormalArgs()
    inner_closure_vars = []
    for i in xrange(len(outer_closure_args)):
      visible_name = "c%d" % i
      name = names.fresh(visible_name)
      args_obj.add_positional(name, visible_name)
      inner_closure_vars.append(Var(name))
    
    data_arg_name = names.fresh("x")
    data_arg_var = Var(data_arg_name)
    idx_arg_name = names.fresh("i")
    idx_arg_var = Var(idx_arg_name)
    
    args_obj.add_positional(data_arg_name, "x")
    args_obj.add_positional(idx_arg_name, "i")
    
    idx_expr = syntax.Index(data_arg_var, idx_arg_var)
    inner_fn = self.get_fn(map_fn)
    fn_call_expr = syntax.Call(inner_fn, tuple(inner_closure_vars)  + (idx_expr,))
    idx_fn = syntax.Fn(name = names.fresh("idx_map"),
                       args = args_obj, 
                       body =  [syntax.Return(fn_call_expr)]
                       )
    
    #t = closure_type.make_closure_type(typed_fn, get_types(closure_args))
    #return Closure(typed_fn, closure_args, t)
    outer_closure_args = tuple(outer_closure_args) + (x,)
  
    idx_closure_t = closure_type.make_closure_type(idx_fn, get_types(outer_closure_args))
    
    idx_closure = Closure(idx_fn, args = outer_closure_args, type = idx_closure_t)
    
    result_type, typed_fn, typed_combine = \
      specialize_IndexReduce(idx_closure, combine, n_indices, init)
    if not self.is_none(init):
      init = self.cast(init, typed_combine.return_type)
    return syntax.IndexReduce(shape = shape, 
                              fn = make_typed_closure(idx_closure, typed_fn),
                              combine = make_typed_closure(combine, typed_combine),
                              init = init,   
                              type = result_type)
    
    
    
  def transform_Reduce(self, expr):
    new_args = self.transform_args(expr.args, flat = True)
    arg_types = get_types(new_args)
    axis = self.transform_if_expr(expr.axis)

    map_fn = self.transform_expr(expr.fn if expr.fn else untyped_identity_function) 
    combine_fn = self.transform_expr(expr.combine)
    
    init = self.transform_expr(expr.init) if expr.init else None
    
    # if there aren't any arrays, just treat this as a function call
    if all(isinstance(t, ScalarT) for t in arg_types):
      return self.invoke(map_fn, new_args)
    
    init_type = init.type if init else None
    
    if self.is_none(axis):
      if adverb_helpers.max_rank(arg_types) > 1:
        assert len(new_args) == 1, \
          "Can't handle multiple reduction inputs and flattening from axis=None"
        #x = new_args[0]
        #return self.flatten_Reduce(map_fn, combine_fn, x, init)
        #new_args = [self.ravel(new_args[0])]
        #arg_types = get_types(new_args)
        
        # Expect that the arguments will get raveled before 
        # the adverb gets evaluated 
        axis = self.none
        arg_types = [array_type.lower_rank(t, t.rank - 1) 
                     for t in arg_types
                     if t.rank > 1]
      else:
        axis = self.int(0)                        
    
    result_type, typed_map_fn, typed_combine_fn = \
        specialize_Reduce(map_fn.type,
                          combine_fn.type,
                          arg_types, 
                          init_type)
    typed_map_closure = make_typed_closure (map_fn, typed_map_fn)
    typed_combine_closure = make_typed_closure(combine_fn, typed_combine_fn)
    
    
    if init_type and init_type != result_type and \
       array_type.rank(init_type) < array_type.rank(result_type):
      assert len(new_args) == 1
      #assert is_zero(axis), "Unexpected axis %s : %s" % (axis, axis.type)
      arg = new_args[0]
      first_elt = syntax.Index(arg, zero_i64, 
                                  type = arg.type.index_type(zero_i64))
      first_combine = specialize(combine_fn, (init_type, first_elt.type))
      first_combine_closure = make_typed_closure(combine_fn, first_combine)
      init = syntax.Call(first_combine_closure, (init, first_elt), 
                                 type = first_combine.return_type)
      slice_rest = syntax.Slice(start = one_i64, stop = none, step = one_i64, 
                                   type = array_type.SliceT(Int64, NoneType, Int64))
      rest = syntax.Index(arg, slice_rest, 
                             type = arg.type.index_type(slice_rest))
      new_args = (rest,)  
    
    return syntax.Reduce(fn = typed_map_closure,
                         combine = typed_combine_closure,
                         args = new_args,
                         axis = axis,
                         type = result_type,
                         init = init)

  def transform_Scan(self, expr):
    map_fn = self.transform_expr(expr.fn if expr.fn else untyped_identity_function)
    combine_fn = self.transform_expr(expr.combine)
    emit_fn = self.transform_expr(expr.emit)
    new_args = self.transform_args(expr.args, flat = True)
    arg_types = get_types(new_args)
    init = self.transform_expr(expr.init) if expr.init else None
    init_type = get_type(init) if init else None
    result_type, typed_map_fn, typed_combine_fn, typed_emit_fn = \
        specialize_Scan(map_fn.type, combine_fn.type, emit_fn.type,
                        arg_types, init_type)
    map_fn.fn = typed_map_fn
    combine_fn.fn = typed_combine_fn
    emit_fn.fn = typed_emit_fn
    axis = self.transform_if_expr(expr.axis)
    if axis is None or self.is_none(axis):
      assert adverb_helpers.max_rank(arg_types) == 1
      axis = syntax_helpers.zero_i64
    return syntax.Scan(fn = make_typed_closure(map_fn, typed_map_fn),
                       combine = make_typed_closure(combine_fn,
                                                    typed_combine_fn),
                       emit = make_typed_closure(emit_fn, typed_emit_fn),
                       args = new_args,
                       axis = axis,
                       type = result_type,
                       init = init)

  def transform_AllPairs(self, expr):
    closure = self.transform_expr(expr.fn)
    new_args = self.transform_args (expr.args, flat = True)
    arg_types = get_types(new_args)
    assert len(arg_types) == 2
    xt,yt = arg_types
    result_type, typed_fn = specialize_AllPairs(closure.type, xt, yt)
    axis = self.transform_if_expr(expr.axis)
    if axis is None or self.is_none(axis):
      axis = zero_i64
    return syntax.AllPairs(fn = make_typed_closure(closure, typed_fn),
                           args = new_args,
                           axis = axis,
                           type = result_type)
  

  
  def infer_phi(self, result_var, val):
    """
    Don't actually rewrite the phi node, just add any necessary types to the
    type environment
    """

    new_val = self.transform_expr(val)
    new_type = new_val.type
    old_type = self.type_env.get(result_var, Unknown)
    new_result_var = self.var_map.lookup(result_var)
    self.type_env[new_result_var]  = old_type.combine(new_type)

  def infer_phi_nodes(self, nodes, direction):
    for (var, values) in nodes.iteritems():
      self.infer_phi(var, direction(values))

  def infer_left_flow(self, nodes):
    return self.infer_phi_nodes(nodes, lambda (x, _): x)

  def infer_right_flow(self, nodes):
    return self.infer_phi_nodes(nodes, lambda (_, x): x)

  
  def transform_phi_node(self, result_var, (left_val, right_val)):
    """
    Rewrite the phi node by rewriting the values from either branch, renaming
    the result variable, recording its new type, and returning the new name
    paired with the annotated branch values
    """

    new_left = self.transform_expr(left_val)
    new_right = self.transform_expr(right_val)
    old_type = self.type_env.get(result_var, Unknown)
    new_type = old_type.combine(new_left.type).combine(new_right.type)
    new_var = self.var_map.lookup(result_var)
    self.type_env[new_var] = new_type
    return (new_var, (new_left, new_right))

  def transform_phi_nodes(self, nodes):
    new_nodes = {}
    for old_k, (old_left, old_right) in nodes.iteritems():
      new_name, (left, right) = self.transform_phi_node(old_k, (old_left, old_right))
      new_nodes[new_name] = (left, right)
    return new_nodes

  def annotate_lhs(self, lhs, rhs_type):

    lhs_class = lhs.__class__
    if lhs_class is syntax.Tuple:
      if rhs_type.__class__ is TupleT:
        assert len(lhs.elts) == len(rhs_type.elt_types)
        new_elts = [self.annotate_lhs(elt, elt_type) 
                    for (elt, elt_type) in zip(lhs.elts, rhs_type.elt_types)]
      else:
        assert rhs_type.__class__ is ArrayT, \
            "Unexpected right hand side type %s for %s" % (rhs_type, lhs)
        elt_type = array_type.lower_rank(rhs_type, 1)
        new_elts = [self.annotate_lhs(elt, elt_type) for elt in lhs.elts]
      tuple_t = tuple_type.make_tuple_type(get_types(new_elts))
      return syntax.Tuple(new_elts, type = tuple_t)
    elif lhs_class is syntax.Index:
      new_arr = self.transform_expr(lhs.value)
      new_idx = self.transform_expr(lhs.index)
      assert isinstance(new_arr.type, ArrayT), \
          "Expected array, got %s" % new_arr.type
      elt_t = new_arr.type.index_type(new_idx.type)
      return syntax.Index(new_arr, new_idx, type = elt_t)
    elif lhs_class is syntax.Attribute:
      name = lhs.name
      struct = self.transform_expr(lhs.value)
      struct_t = struct.type
      assert isinstance(struct_t, StructT), \
          "Can't access fields on value %s of type %s" % \
          (struct, struct_t)
      field_t = struct_t.field_type(name)
      return syntax.Attribute(struct, name, field_t)
    else:
      assert lhs_class is Var, "Unexpected LHS: %s" % (lhs,)
      new_name = self.var_map.lookup(lhs.name)
      old_type = self.type_env.get(new_name, Unknown)
      new_type = old_type.combine(rhs_type)
      self.type_env[new_name] = new_type
      return Var(new_name, type = new_type)

  def transform_Assign(self, stmt):
    rhs = self.transform_expr(stmt.rhs)
    lhs = self.annotate_lhs(stmt.lhs, rhs.type)
    return syntax.Assign(lhs, rhs)
  
  
  
  def transform_If(self, stmt):
    cond = self.transform_expr(stmt.cond) 

    assert isinstance(cond.type, ScalarT), \
        "Condition %s has type %s but must be convertible to bool" % (cond, cond.type)
    # it would be cleaner to not have anything resembling an optimization 
    # inter-mixed with the type inference, but I'm not sure how else to 
    # support 'if x is None:...'
    if is_true(cond):
      self.blocks.top().extend(self.transform_block(stmt.true))
      for (name, (left,_)) in stmt.merge.iteritems():
        typed_left = self.transform_expr(left)
        typed_var = self.annotate_lhs(Var(name), typed_left.type) 
        self.assign(typed_var, typed_left)
      return
    
    if is_false(cond):
      self.blocks.top().extend(self.transform_block(stmt.false))
      for (name, (_,right)) in stmt.merge.iteritems():
        typed_right = self.transform_expr(right)
        typed_var = self.annotate_lhs(Var(name), typed_right.type)
        self.assign(typed_var, typed_right)
      return
    true = self.transform_block(stmt.true)
    false = self.transform_block(stmt.false) 
    merge = self.transform_phi_nodes(stmt.merge)
    return syntax.If(cond, true, false, merge)

  def transform_Return(self, stmt):
    ret_val = self.transform_expr(stmt.value)
    curr_return_type = self.type_env["$return"]
    self.type_env["$return"] = curr_return_type.combine(ret_val.type)
    return syntax.Return(ret_val)

  def transform_While(self, stmt):
    self.infer_left_flow(stmt.merge)
    cond = self.transform_expr(stmt.cond)
    body = self.transform_block(stmt.body)
    merge = self.transform_phi_nodes(stmt.merge)
    return syntax.While(cond, body, merge)

  def transform_ForLoop(self, stmt):
    self.infer_left_flow(stmt.merge)
    start = self.transform_expr(stmt.start)
    stop = self.transform_expr(stmt.stop)
    step = self.transform_expr(stmt.step)
    lhs_t = start.type.combine(stop.type).combine(step.type)
    var = self.annotate_lhs(stmt.var, lhs_t)
    body = self.transform_block(stmt.body)
    merge = self.transform_phi_nodes(stmt.merge)
    return syntax.ForLoop(var, start, stop, step, body, merge)



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

  try: 
    tenv = typed_args.bind(types,
                           keyword_fn = keyword_fn,
                           starargs_fn = tuple_type.make_tuple_type)
  except  MissingArgsError as e:
    e.fn_name = untyped_fn.name 
    raise e 
    
  except: 
    print "Error while calling %s with types %s" % (untyped_fn, types)
    raise
  # keep track of the return
  tenv['$return'] = Unknown
  annotator = Annotator(tenv, var_map)
  body = annotator.transform_block(untyped_fn.body)
  arg_names = [local_name for local_name
               in
               typed_args.nonlocals + tuple(typed_args.positional)
               if local_name not in unbound_keywords]
  if len(unbound_keywords) > 0:
    default_assignments = []
    for local_name in unbound_keywords:
      t = tenv[local_name]
      python_value = typed_args.defaults[local_name]
      var = Var(local_name, type = t)
      if isinstance(python_value, tuple):
        parakeet_elts = []
        for (elt_value, elt_type) in zip(python_value, t.elt_types):
          parakeet_elt = syntax.Const(elt_value, elt_type)
          parakeet_elts.append(parakeet_elt)
        typed_val = syntax.Tuple(tuple(parakeet_elts), type = t)
      else:
        typed_val = syntax.Const(python_value, t) #mk_default_const(python_value, t)

      stmt = syntax.Assign(var, typed_val)
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
  if isinstance(return_type,  UnknownT):
    body.append(syntax.Return(syntax_helpers.none))
    tenv["$return"] = NoneType
    return_type = NoneType

  return syntax.TypedFn(
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

  if isinstance(fn, syntax.TypedFn):
    return fn
  typed_fundef = infer_types(fn, arg_types)
  from rewrite_typed import rewrite_typed
  coerced_fundef = rewrite_typed(typed_fundef, return_type)
  import simplify
  normalized = simplify.Simplify().apply(coerced_fundef)
  return normalized

def _get_fundef(fn):
  if isinstance(fn, (syntax.Fn, syntax.TypedFn)):
    return fn
  else:
    assert isinstance(fn, str), \
        "Unexpected function %s : %s"  % (fn, fn.type)
    return syntax.Fn.registry[fn]

def _get_closure_type(fn):
  assert isinstance(fn, (Fn, TypedFn, ClosureT, Closure, Var)), \
    "Expected function, got %s" % fn
    
  if fn.__class__ is closure_type.ClosureT:
    return fn
  elif isinstance(fn, Closure):
    return fn.type
  elif isinstance(fn, Var):
    assert isinstance(fn.type, closure_type.ClosureT)
    return fn.type
  else:
    fundef = _get_fundef(fn)
    return closure_type.make_closure_type(fundef, [])

def specialize(fn, arg_types, return_type = None):
  if config.print_before_specialization:
    if return_type:
      print "==== Specializing", fn, "for input types", arg_types, "and return type", return_type
    else:  
      print "=== Specializing", fn, "for types", arg_types 
  if isinstance(fn, syntax.TypedFn):
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

  typed = specialize(untyped, arg_types)
  return typed.return_type

def specialize_IndexMap(fn, n_indices):
  idx_type = make_tuple_type( (Int64,) * n_indices) if n_indices > 1 else Int64
  typed_fn = specialize(fn, (idx_type,))
  result_type = array_type.increase_rank(typed_fn.return_type, n_indices)
  return result_type, typed_fn

def specialize_IndexReduce(fn, combine, n_indices, init = None):
  idx_type = make_tuple_type( (Int64,) * n_indices) if n_indices > 1 else Int64
  if init is None or isinstance(init.type, NoneT):
    typed_fn = specialize(fn, (idx_type,))
  else:
    typed_fn = specialize(fn, (idx_type,), return_type = init.type)
  elt_type = typed_fn.return_type
  typed_combine = specialize(combine, (elt_type, elt_type))
  return elt_type, typed_fn, typed_combine 
      
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
    acc_type = init_type

  typed_combine_fn = specialize(combine_fn, [acc_type, elt_type])
  new_acc_type = typed_combine_fn.return_type
  if new_acc_type != acc_type:
    typed_combine_fn = specialize(combine_fn, [new_acc_type, elt_type])
    new_acc_type = typed_combine_fn.return_type
  #assert new_acc_type == acc_type, \
  #  "Expected accumulator types %s but encountered %s" % (acc_type, new_acc_type)
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
