import transform
 
import array_type 
import core_types 
import syntax

class LowerIndexing(transform.Transform):
  def transform_Index(self, expr):
    """
    For now assume we're only dealing with indexing by a single scalar
    """
    arr = self.transform_expr(expr.value)
    arr_t = arr.type 
    assert isinstance(arr_t, array_type.ArrayT), "Unexpected array %s : %s" % (arr, arr.type) 
    idx = self.transform_expr(expr.index)
    
    data_ptr = self.attr(arr, "data")
    idx_t = idx.type
    if isinstance(idx_t, core_types.IntT):
      stride0 = self.strides(arr, 0)
      offset_bytes = self.mul(stride0, idx, "offset_bytes")
      elt_size = arr_t.elt_type.dtype.itemsize
      offset_elts = self.div(offset_bytes, elt_size, "offset_elt")
    else:
      offset_elts = self.zero_i64("offset")
      for i in xrange(len(idx_t.elt_types)):
        idx_i = syntax.TupleProj(idx, syntax.Const(i, type=core_types.Int64))
        stride = self.strides(arr, i)
        offset_bytes = self.mul(stride, idx_i, "offset_bytes")
        elt_size = arr_t.elt_type.dtype.itemsize
        offset_elts = self.add(offset_elts, self.div(offset_bytes, elt_size, "offset_elt"))
    return self.index(data_ptr, offset_elts, temp = False)
   
def lower_indexing(fn):
  return transform.cached_apply(LowerIndexing, fn)
  

