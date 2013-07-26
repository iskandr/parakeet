from .. import names
from ..ndtypes import (TupleT, ScalarT, IntT, Int64, ArrayT, 
                       combine_type_list, increase_rank, 
                       make_array_type, make_tuple_type, make_slice_type, 
                       type_conv)
from ..syntax import (Array, AllocArray, Attribute, Cast, Const, Expr, Index, 
                      Range, Ravel, Reshape, Slice, TupleProj, Var)
from ..syntax.helpers import get_types 
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
    assert result.type is not None,  \
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
    elt_types = get_types(new_elts)
    common_t = combine_type_list(elt_types)
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
  

  
  def transform_Range(self, expr):
    start = self.transform_expr(expr.start) if expr.start else None
    stop = self.transform_expr(expr.stop) if expr.stop else None
    step = self.transform_expr(expr.step) if expr.step else None
    array_t = ArrayT(Int64, 1)
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
    #elt_types = get_types(elts)
    #t = tuple_type.make_tuple_type(elt_types)
    #return syntax.Tuple(elts, type = t)

  def transform_Const(self, expr):
    return Const(expr.value, type_conv.typeof(expr.value))
  


  def transform_Reshape(self, expr):
    array = self.transform_expr(expr.array)
    shape = self.transform_expr(expr.shape)
    rank = len(shape.type.elt_types)
    assert array.type.__class__ is ArrayT
    t = make_array_type(array.elt_type, rank)
    return Reshape(array, shape, type = t)
  
  def transform_Ravel(self, expr):
    array = self.transform_expr(expr.array)
    if array.type.__class__ is not  ArrayT:
      print "Warning: Can't ravel/flatten an object of type %s" % array.type 
      return array 
    t = make_array_type(array.type.elt_type, 1)
    return Ravel(array, type = t)
  
  
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
      assert t.__class__ is TupleT, \
         "Unexpected argument type for 'len': %s" % t
      return Const(len(t.elt_types), type = Int64)