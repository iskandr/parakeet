



 

from .. import  syntax

from .. ndtypes import (ScalarT, ArrayT, make_array_type, TupleT, 
                        Int32, Int64,  ptr_type, PtrT, closure_type)  
from .. syntax import Struct, Assign, Const, Index, Attribute, Var, Tuple, Alloc  
from .. syntax.helpers import const_int, const_tuple, zero
from transform import Transform


class LowerStructs(Transform):
  """The only non-scalar objects should all be created as explicit Structs"""

  def transform_TypedFn(self, expr):
    import pipeline
    return pipeline.lowering.apply(expr)

  def transform_Tuple(self, expr):
    struct_args = self.transform_expr_list(expr.elts)
    return Struct(struct_args, type = expr.type)

  def transform_TupleProj(self, expr):
    new_tuple = self.transform_expr(expr.tuple)
    assert isinstance(expr.index, int)
    tuple_t = expr.tuple.type
    field_name, field_type  = tuple_t._fields_[expr.index]
    return Attribute(new_tuple, field_name, type = field_type)

  def transform_Slice(self, expr):
    struct_args = self.transform_expr_list([expr.start, expr.stop, expr.step])
    return Struct(struct_args, type = expr.type)

  def transform_Assign(self, stmt):
    lhs, rhs = stmt.lhs, stmt.rhs
    if isinstance(lhs, Tuple):
      for (i, lhs_elt) in enumerate(lhs.elts):
        self.assign(lhs_elt, self.tuple_proj(rhs, i), recursive = True)
    else:
      assert isinstance(lhs, (Var, Index, Attribute)), \
          "Invalid LHS: %s" % (lhs,)
      return Assign(stmt.lhs, self.transform_expr(rhs))

  def transform_Closure(self, expr):
    _ = self.transform_expr(expr.fn)
    closure_args = self.transform_expr_list(expr.args)
    closure_id = closure_type.id_of_closure_type(expr.type)
    closure_id_node = Const(closure_id, type = Int64)
    return Struct([closure_id_node] + closure_args, type = expr.type)

  def transform_ClosureElt(self, expr):
    new_closure = self.transform_expr(expr.closure)
    assert isinstance(expr.index, int)
    # first field is always the closure ID, so we have to
    # index 1 higher
    field_name, field_type = new_closure.type._fields_[expr.index + 1]
    return Attribute(new_closure, field_name, type = field_type)

  def array_view(self, data, shape, strides, offset, nelts):
    """Helper function used by multiple array-related transformations"""

    data = self.assign_name(self.transform_expr(data), "data_ptr")
    data_t = data.type
    assert isinstance(data_t, PtrT), \
        "Data field of array must be a pointer, got %s" % data_t

    shape = self.assign_name(self.transform_expr(shape), "shape")
    shape_t = shape.type
    assert isinstance(shape_t, TupleT), \
        "Shape of array must be a tuple, got: %s" % shape_t

    strides = self.assign_name(self.transform_expr(strides), "strides")
    strides_t = strides.type
    assert isinstance(strides_t, TupleT), \
        "Strides of array must be a tuple, got: %s" % strides_t

    rank = len(shape_t.elt_types)
    strides_rank = len(strides_t.elt_types)
    assert rank == strides_rank, \
        "Shape and strides must be of same length, but got %d and %d" % \
        (rank, strides_rank)
    array_t = make_array_type(data_t.elt_type, rank)
    return Struct([data, shape, strides, offset, nelts], type = array_t)

  def transform_ArrayView(self, expr):
    array_struct = self.array_view(expr.data, expr.shape, expr.strides,
                                   expr.offset, expr.size)
    assert expr.type == array_struct.type, \
        "Mismatch between original type %s and transformed type %s" % \
        (expr.type, array_struct.type)
    return array_struct

  def transform_Array(self, expr):
    """Array literal"""

    array_t = expr.type
    assert isinstance(array_t, ArrayT)
    elt_t = array_t.elt_type
    assert isinstance(elt_t, ScalarT)

    elts = self.transform_expr_list(expr.elts)
    n = len(elts)

    ptr_t = ptr_type(elt_t)
    alloc = Alloc(elt_t, const_int(n), type = ptr_t)
    ptr_var = self.assign_name(alloc, "data")

    for (i, elt) in enumerate(elts):
      idx = Const(i, type = Int32)
      lhs = Index(ptr_var, idx, type = elt_t)
      self.assign(lhs, elt)

    return self.array_view(ptr_var, const_tuple(n), const_tuple(1),
                           offset = const_int(0), nelts = const_int(n))


  def transform_Range(self, expr):
    diff = self.sub(expr.stop, expr.start, "range_diff")
    nelts = self.safediv(diff, expr.step, name="nelts_raw")
    result = self.alloc_array(Int64, 
                              (nelts,), 
                              name = "range_result", 
                              explicit_struct = True)
    ptr = self.attr(result, "data", "data_ptr")
    def loop_body(i):
      v = self.add(expr.start, self.mul(i, expr.step)) 
      self.setidx(ptr, i, v)
    self.loop(zero(Int64), nelts, loop_body, return_stmt = False, while_loop = False)
    return result 
  
  def transform_ConstArray(self, expr):
    assert False, "ConstArray not implemented"
    
  def transform_Ravel(self, expr):
    return self.ravel(self.transform_expr(expr.array), explicit_struct = True)