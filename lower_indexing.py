import transform
 
import array_type 
import core_types 

class LowerIndexing(transform.Transform):
  def transform_Index(self, expr):
    """
    For now assume we're only dealing with indexing by a single scalar
    """
    arr = self.transform_expr(expr.value)
    assert isinstance(arr.type, array_type.ArrayT), "Unexpected array %s : %s" % (arr, arr.type) 
    idx = self.transform_expr(expr.index)
    assert isinstance(idx.type, core_types.IntT)

    strides = self.get_struct_field(arr, "strides")
    stride0 = self.get_index(strides, 0)
    offset_bytes = self.mul(stride0, idx, "offset_bytes")
    elt_size = arr.type.elt_type.dtype.itemsize
    offset_elts = self.div(offset_bytes, elt_size, "offset_elt")
    data_ptr = self.get_struct_field(arr, "data")
    result = self.get_index(data_ptr, offset_elts)
    return result 
    

def lower_indexing(fn):
  return transform.cached_apply(LowerIndexing, fn)
  

