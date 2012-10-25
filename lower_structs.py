import syntax
import core_types 

from array_type import ArrayT, ScalarT
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
  
  def transform_Array(self, expr):

    n = len(expr.elts)    
    array_t = expr.type
    assert isinstance(array_t, ArrayT)

    elt_t = array_t.elt_type
    assert isinstance(elt_t, ScalarT)
    ptr_t = core_types.ptr_type(elt_t)
    alloc = syntax.Alloc(elt_t, const_int(n), type = ptr_t)
    ptr_var = self.fresh_var(ptr_t, "data")
    self.insert_stmt(syntax.Assign(ptr_var, alloc))
    for (i, elt) in enumerate(self.transform_expr_list(expr.elts)):
 
      idx = syntax.Const(i, type = core_types.Int32)
      lhs = syntax.Index(ptr_var, idx, type = elt_t)
      self.insert_assign(lhs, elt)
    
    shape = self.transform_Tuple(const_tuple(n))
    shape_var = self.fresh_var(shape.type, "shape")
    self.insert_assign(shape_var, shape)
    strides = self.transform_Tuple(const_tuple(elt_t.nbytes))
    strides_var = self.fresh_var(strides.type, "strides")
    self.insert_assign(strides_var, strides)
    return syntax.Struct([ptr_var, shape_var, strides_var], type = expr.type)
  
def make_structs_explicit(fn):
  return cached_apply(LowerStructs, fn)