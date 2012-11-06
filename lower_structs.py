import syntax
import core_types 

from array_type import ArrayT, ScalarT, make_array_type

import tuple_type 
import closure_signatures 
from transform import Transform, cached_apply

from syntax_helpers import const_tuple, const_int 


class LowerStructs(Transform):
  """
  The only non-scalar objects should all be created as explicit Structs
  """  
  def transform_Tuple(self, expr):
    struct_args = self.transform_expr_list(expr.elts)
    return syntax.Struct(struct_args, type = expr.type)
    
  
  def transform_Assign(self, stmt):
    lhs, rhs = stmt.lhs, stmt.rhs 
    if isinstance(lhs, syntax.Tuple):
      for (i, lhs_elt) in enumerate(lhs.elts):
        self.assign(lhs_elt, self.tuple_proj(rhs, i), recursive = True)
    else:
      assert isinstance(lhs, (syntax.Var, syntax.Index))
      return syntax.Assign(stmt.lhs, self.transform_expr(rhs))
        
  def transform_Closure(self, expr):
    closure_args = self.transform_expr_list(expr.args)
    closure_id = closure_signatures.get_id(expr.type)
    closure_id_node = syntax.Const(closure_id, type = core_types.Int64)
    return syntax.Struct([closure_id_node] + closure_args, type = expr.type)

  def transform_TupleProj(self, expr):
    new_tuple = self.transform_expr(expr.tuple)
    assert isinstance(expr.index, int)
    tuple_t = expr.tuple.type

    field_name, field_type  = tuple_t._fields_[expr.index]
    return syntax.Attribute(new_tuple, field_name, type = field_type)
  
  def array_view(self, data, shape, strides):
    """
    Helper function used by multiple array-related transformations
    """
    data = self.assign_temp(self.transform_expr(data), "data_ptr")
    data_t = data.type 
    assert isinstance(data_t, core_types.PtrT), \
      "Data field of array must be a pointer, got %s" % data_t 
      
    shape = self.assign_temp(self.transform_expr(shape), "shape")
    shape_t = shape.type
    assert isinstance(shape_t, tuple_type.TupleT), \
      "Shape of array must be a tuple, got: %s" % shape_t
    
    strides = self.assign_temp(self.transform_expr(strides), "strides")
    strides_t = strides.type  
    assert isinstance(strides_t, tuple_type.TupleT), \
      "Strides of array must be a tuple, got: %s" % strides_t
    
    rank = len(shape_t.elt_types)
    strides_rank = len(strides_t.elt_types)
    assert rank == strides_rank, \
      "Shape and strides must be of same length, but got %d and %d" % \
      (rank, strides_rank) 
    array_t = make_array_type(data_t.elt_type, rank)
    return syntax.Struct([data, shape, strides], type = array_t)
    
  def transform_ArrayView(self, expr):
    array_struct = self.array_view(expr.data, expr.shape, expr.strides)
    assert expr.type == array_struct.type, \
      "Mismatch between original type %s and transformed type %s" % \
      (expr.type, array_struct.type)
    return array_struct 
  
      
  def transform_Array(self, expr):
    """
    Array literal
    """
    n = len(expr.elts)    
    array_t = expr.type
    assert isinstance(array_t, ArrayT)

    elt_t = array_t.elt_type
    assert isinstance(elt_t, ScalarT)
    ptr_t = core_types.ptr_type(elt_t)
    alloc = syntax.Alloc(elt_t, const_int(n), type = ptr_t)
    ptr_var = self.assign_temp(alloc, "data")
    
    for (i, elt) in enumerate(self.transform_expr_list(expr.elts)):
      idx = syntax.Const(i, type = core_types.Int32)
      lhs = syntax.Index(ptr_var, idx, type = elt_t)
      self.assign(lhs, elt)
    
    return self.array_view(ptr_var, const_tuple(n), const_tuple(1))
    
def make_structs_explicit(fn):
  return cached_apply(LowerStructs, fn)