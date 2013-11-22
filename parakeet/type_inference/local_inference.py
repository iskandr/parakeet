from .. import names
from ..ndtypes import (StructT, Type, Unknown,
                       ArrayT, TypeValueT, TupleT,
                       ScalarT, IntT, Int64,  Bool, Float64,    
                       combine_type_list, increase_rank, lower_rank,  
                       make_array_type, make_tuple_type, make_slice_type, make_closure_type, 
                       repeat_tuple, 
                       type_conv, )

from ..syntax import (Array, AllocArray, Attribute, 
                      Cast, Closure, Const, ConstArray, ConstArrayLike, 
                      Expr, Index, 
                      Range, Ravel, Reshape, Shape,
                      Select, Slice, 
                      Transpose, Tuple, TupleProj, TypeValue,  Var,  
                      ForLoop, While, Assign, Return, If)

from ..syntax.helpers import get_types, is_true, is_false, const_int
from ..syntax.wrappers import build_untyped_prim_fn

from ..transforms import Transform


class LocalTypeInference(Transform):
  """
  Local type inference information which doesn't 
  require recursive calls back into the inference 
  algorithm. 
  """
  def __init__(self, tenv, var_map):
    Transform.__init__(self)
    self.type_env = tenv 
    self.var_map = var_map
    
  def transform_expr(self, expr):
    from ..frontend import ast_conversion
    if not isinstance(expr, Expr):
      expr = ast_conversion.value_to_syntax(expr)
    result = Transform.transform_expr(self, expr)  
    assert result.type is not None, \
       "Unsupported expression encountered during type inference: %s" % (expr,) 
    return result 


  def transform_Index(self, expr):
    value = self.transform_expr(expr.value)
    index = self.transform_expr(expr.index)
    if isinstance(value.type, TupleT):
      assert isinstance(index.type, IntT)
      assert index.__class__  is Const
      i = index.value
      assert isinstance(i, int)
      elt_types = value.type.elt_types
      assert i < len(elt_types), \
          "Can't get element %d of length %d tuple %s : %s" % \
          (i, len(elt_types), value, value.type)
      elt_t = value.type.elt_types[i]
      return TupleProj(value, i, type = elt_t)
    else:
      result_type = value.type.index_type(index.type)
      return Index(value, index, type = result_type)

  
  def transform_Array(self, expr):
    
    new_elts = self.transform_args(expr.elts)
    if expr.type is None:
      elt_types = get_types(new_elts)
      if len(elt_types) > 0:
        common_t = combine_type_list(elt_types)
        if common_t is Unknown:
          raise TypeError("Couldn't find commom type for elements of array %s" % expr)
      else:
        # numpy defaults to Float arrays 
        common_t = Float64 
    else:
      common_t = expr.type 
    array_t = increase_rank(common_t, 1)
    return Array(new_elts, type = array_t)

  def transform_AllocArray(self, expr):
    elt_type = expr.elt_type
    assert isinstance(elt_type, ScalarT), \
      "Invalid array element type  %s" % (elt_type)
      
    shape = self.transform_expr(expr.shape)
    if isinstance(shape, ScalarT):
      shape = self.cast(shape, Int64)
      shape = self.tuple((shape,), "array_shape")
    assert isinstance(shape, TupleT), \
      "Invalid shape %s" % (shape,)
    rank = len(shape.elt_types)
    t = make_array_type(elt_type, rank)
    return AllocArray(shape, elt_type, type = t)

  def transform_ConstArray(self, expr):
    shape = self.transform_expr(expr.shape)
    value = self.transform_expr(expr.value)
    assert isinstance(value.type, ScalarT), \
      "ConstArray expects scalar value, got %s: %s" % (value, value.type)
    if isinstance(shape.type, ScalarT):
      shape = self.tuple([shape])
    assert isinstance(shape.type, TupleT), \
      "Expected shape of ConstArray to be tuple, got %s : %s" % (shape, shape.type)
    ndims = len(shape.type.elt_types)
    if ndims == 0:
      return value 
    array_t = make_array_type(value.type, ndims)
    return ConstArray(shape, value, type = array_t) 
  
  def transform_ConstArrayLike(self, expr):
    array = self.transform_expr(expr.array)
    value = self.transform_expr(expr.value)
    assert isinstance(value.type, ScalarT), \
      "ConstArray expects scalar value, got %s: %s" % (value, value.type)
    if isinstance(array.type, ScalarT):
      return value 
    elif isinstance(array.type, TupleT):
      array = self.transform_expr(Array(elts = self.tuple_elts(array)))
      
    assert isinstance(array.type, ArrayT), \
      "ConstArrayLike expected array, got %s : %s" % (array, array.type)
    shape = self.shape(array)
    ndims = len(shape.type.elt_types)
    if ndims == 0:
      return value 
    array_t = make_array_type(value.type, ndims)
    return ConstArrayLike(array = array, value = value, type = array_t)
    

  
  def transform_Range(self, expr):
    start = self.transform_expr(expr.start) if expr.start else None
    stop = self.transform_expr(expr.stop) if expr.stop else None
    step = self.transform_expr(expr.step) if expr.step else None
    
    # by default we're generating ranges of Int64
    # but also allow for floating values 
    elt_t = Int64 
    if not self.is_none(start): elt_t = elt_t.combine(start.type)
    if not self.is_none(stop): elt_t = elt_t.combine(stop.type)
    if not self.is_none(step): elt_t = elt_t.combine(step.type)
    array_t = make_array_type(elt_t, 1)
    return Range(start, stop, step, type = array_t)

  def transform_Slice(self, expr):
    start = self.transform_expr(expr.start)
    stop = self.transform_expr(expr.stop)
    step = self.transform_expr(expr.step)
    slice_t = make_slice_type(start.type, stop.type, step.type)
    return Slice(start, stop, step, type = slice_t)


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

  def transform_Const(self, expr):
    if expr.type is None:
      t = type_conv.typeof(expr.value)
    else:
      t = expr.type 
    return Const(expr.value, type = t)
  

  def transform_Shape(self, expr):
    array = self.transform_expr(expr.array)
    if isinstance(array.type, ArrayT):
      t = repeat_tuple(Int64, array.type.rank)
      return Shape(array = array, type = t)
    elif isinstance(array.type, TupleT):
      elt = const_int(len(array.type.elt_types))
      return self.tuple( [elt]  )
    else:
      return self.tuple( [] )
      

  def transform_Reshape(self, expr):
    array = self.transform_expr(expr.array)
    shape = self.transform_expr(expr.shape)
    rank = len(shape.type.elt_types)
    assert array.type.__class__ is ArrayT
    t = make_array_type(array.elt_type, rank)
    return Reshape(array, shape, type = t)
  
  def transform_Ravel(self, expr):
    array = self.transform_expr(expr.array)
    if isinstance(array.type, ScalarT):
      return array 
    assert array.type.__class__ is ArrayT, \
      "Can't ravel/flatten %s of type %s" % (array, array.type) 
    t = make_array_type(array.type.elt_type, 1)
    return Ravel(array, type = t)
  
  def transform_Transpose(self, expr):
    array = self.transform_expr(expr.array)
    if isinstance(array.type, ScalarT):
      return array
    assert array.type.__class__ is ArrayT, \
      "Can't transpose %s of type %s" % (array, array.type) 
    return Transpose(array = array, type = array.type)
  
  def transform_Cast(self, expr):
    v = self.transform_expr(expr.value)
    return Cast(v, type = expr.type)
  
  def transform_Len(self, expr):
    v = self.transform_expr(expr.value)
    t = v.type
    if t.__class__ is ArrayT:
      shape_t = make_tuple_type([Int64] * t.rank)
      shape = Attribute(v, 'shape', type = shape_t)
      return TupleProj(shape, 0, type = Int64)
    else:
      assert t.__class__ is TupleT, "Unexpected argument for 'len' - %s : %s" % (expr.value, t)
      return Const(len(t.elt_types), type = Int64)

  def transform_DelayUntilTyped(self, expr):
    new_values = self.transform_expr_tuple(expr.values)
    if expr.keywords:
      typed_keywords = {}
      for k, v in expr.keywords.iteritems():
        typed_keywords[k] = self.transform_expr(v)
      new_syntax = expr.fn(*new_values, **typed_keywords)
    else:
      new_syntax = expr.fn(*new_values)
    assert new_syntax.type is not None, \
      "Error in %s, new expression %s lacks type" % (expr, new_syntax)
    return new_syntax
  
  def transform_TypeValue(self, expr):
    t = expr.type_value 
    assert isinstance(t, Type), "Invalid type value %s" % (t,)
    return TypeValue(t, type=TypeValueT(t))
    
  def transform_Closure(self, expr):
    new_args = self.transform_expr_list(expr.args)
    t = make_closure_type(expr.fn, get_types(new_args))
    return Closure(expr.fn, new_args, type = t)

  def transform_Arith(self, expr):
    return build_untyped_prim_fn(expr)
    #t = make_closure_type(untyped_fn, ())
    #return Closure(untyped_fn, (), type = t)

  def transform_UntypedFn(self, expr):
    return expr 

  def transform_Attribute(self, expr):
    value = self.transform_expr(expr.value)
    assert isinstance(value.type, StructT)
    result_type = value.type.field_type(expr.name)
    return Attribute(value, expr.name, type = result_type)
  
  def transform_Select(self, expr):
    trueval = self.transform_expr(expr.true_value)
    falseval = self.transform_expr(expr.false_value)
    t = trueval.type.combine(falseval.type)
    trueval = self.cast(trueval, t)
    falseval = self.cast(falseval, t) 
    cond = self.transform_expr(expr.cond)
    cond = self.cast(cond, Bool)
    return Select(cond = cond, true_value = trueval, false_value = falseval, type = t)
  
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
    #
    # Am I a tuple of Assignments? 
    #
    if lhs_class is Tuple:
      if rhs_type.__class__ is TupleT:
        assert len(lhs.elts) == len(rhs_type.elt_types)
        new_elts = [self.annotate_lhs(elt, elt_type) 
                    for (elt, elt_type) 
                    in zip(lhs.elts, rhs_type.elt_types)]
      else:
        assert rhs_type.__class__ is ArrayT, \
            "Unexpected right hand side type %s for %s" % (rhs_type, lhs)
        elt_type = lower_rank(rhs_type, 1)
        new_elts = [self.annotate_lhs(elt, elt_type) for elt in lhs.elts]
      tuple_t = make_tuple_type(get_types(new_elts))
      return Tuple(new_elts, type = tuple_t)
    
    #
    # Index Assignment
    #
    elif lhs_class is Index:
      new_arr = self.transform_expr(lhs.value)
      new_idx = self.transform_expr(lhs.index)
      assert new_arr.type.__class__ is ArrayT, \
          "Expected array, got %s" % new_arr.type
      elt_t = new_arr.type.index_type(new_idx.type)
      return Index(new_arr, new_idx, type = elt_t)
    
    # 
    # Attribute Assignment for Mutable Objects
    # 
    elif lhs_class is Attribute:
      name = lhs.name
      struct = self.transform_expr(lhs.value)
      struct_t = struct.type
      assert isinstance(struct_t, StructT), \
          "Can't access fields on value %s of type %s" % \
          (struct, struct_t)
      field_t = struct_t.field_type(name)
      return Attribute(struct, name, field_t)
    
    # 
    # Regular Binding of Names to Values
    # 
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
    return Assign(lhs, rhs)
  
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
    return If(cond, true, false, merge)

  def transform_Return(self, stmt):
    ret_val = self.transform_expr(stmt.value)
    curr_return_type = self.type_env["$return"]
    self.type_env["$return"] = curr_return_type.combine(ret_val.type)
    return Return(ret_val)

  def transform_While(self, stmt):
    self.infer_left_flow(stmt.merge)
    cond = self.transform_expr(stmt.cond)
    body = self.transform_block(stmt.body)
    merge = self.transform_phi_nodes(stmt.merge)
    return While(cond, body, merge)

  def transform_ForLoop(self, stmt):
    start = self.transform_expr(stmt.start)
    stop = self.transform_expr(stmt.stop)
    step = self.transform_expr(stmt.step)
    lhs_t = start.type.combine(stop.type).combine(step.type)
    var = self.annotate_lhs(stmt.var, lhs_t)
    self.infer_left_flow(stmt.merge)
    body = self.transform_block(stmt.body)
    merge = self.transform_phi_nodes(stmt.merge)
    return ForLoop(var, start, stop, step, body, merge)

