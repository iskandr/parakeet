from ..syntax import zero_i64, one_i64, Alloc, const_int, const_tuple,  Index, ArrayView 
from ..ndtypes import (ScalarT, PtrT, TupleT, ArrayT, ptr_type, Int64)
from transform import Transform 

class ExplicitArrayAlloc(Transform):

    
  
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
      lhs = Index(ptr_var, const_int(i), type = elt_t)
      self.assign(lhs, elt)

    return self.array_view(ptr_var, const_tuple(n), const_tuple(1),
                           offset = const_int(0), nelts = const_int(n))