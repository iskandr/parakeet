import transform
 
import array_type 
import core_types 

class LowerIndexing(transform.Transform):
  
  
  
  def transform_lhs_Index(self, expr):
    """
    For now assume we're only dealing with indexing by a single scalar
    """
    arr = self.transform_expr(expr.value)
    arr_t = arr.type 
    assert isinstance(arr_t, array_type.ArrayT), "Unexpected array %s : %s" % (arr, arr.type) 
    idx = self.assign_temp(self.transform_expr(expr.index), "idx")
    
    data_ptr = self.attr(arr, "data")
    idx_t = idx.type
    if isinstance(idx_t, core_types.IntT):
      stride0 = self.strides(arr, 0)
      offset_elts= self.mul(stride0, idx, "offset_elts")
    else:
      offset_elts = self.zero_i64("offset")
      for i in xrange(len(idx_t.elt_types)):
        idx_i = self.assign_temp(self.tuple_proj(idx, i), "idx_%d" % i)
        stride = self.strides(arr, i)
        elts_i = self.mul(stride, idx_i, "offset_bytes")
        offset_elts = self.add(offset_elts, elts_i)
    return self.index(data_ptr, offset_elts, temp = False)
  
  def transform_Index(self, expr):
    return self.assign_temp(self.transform_lhs_Index(expr),"idx_result")
   
def lower_indexing(fn):
  return transform.cached_apply(LowerIndexing, fn)
  

