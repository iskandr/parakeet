import transform
 
import array_type 
import core_types 
import tuple_type
import syntax 
import syntax_helpers
class LowerIndexing(transform.Transform):
  
  
  def process_index(self, data_ptr, shape, strides, idx, i):
    if 
      
      
      
  def array_slice(self, arr, indices):
    data_ptr = self.attr(arr, "data") 
    shape = self.attr(arr, "shape")
    strides = self.attr(arr, "strides")
    elt_size = syntax_helpers.const_int(arr.nbytes(), core_types.Int64)
    
    new_strides = []
    new_shape = []
    elt_offset = syntax_helpers.zero_i64
    n_indices = enumerate(indices)
    for (i, idx) in enumerate(indices):
      stride_i = self.strides(arr, i)
      shape_i = self.shape(arr, i)
      if syntax_helpers.is_scalar(idx):
        elt_offset = self.mul(idx, stride_i, "elt_offset")
        byte_offset = self.mul(elt_offset, elt_size)
        data_ptr = self.incr_ptr(data_ptr, byte_offset)
        shape_
      else:
        final_rank += 1
      data_ptr, shape, strides = self.process_index(data_ptr, shape, strides, idx, i) 
    
    return data_ptr, shape, strides
      
  def transform_lhs_Index(self, expr):
    arr = self.transform_expr(expr.value)
    arr_t = arr.type
    assert isinstance(arr_t, array_type.ArrayT), "Unexpected array %s : %s" % (arr, arr.type) 
    idx = self.assign_temp(self.transform_expr(expr.index), "idx")
    idx_t = idx.type    
    if isinstance(idx_t, tuple_type.TupleT):
      n_elts = len(idx_t.elt_types)
      indices = [self.assign_temp(self.tuple_proj(idx, i), "idx_%d" % i) 
                 for i in xrange(n_elts)]
    else:
      indices = [idx]
    # right pad the index expression with None for each missing index
    n_given = len(indices)
    n_required = arr_t.rank 
    if n_given < n_required:
      extra_indices = [syntax_helpers.const_none] * (n_required - n_given)
      indices.extend(extra_indices)
    
    # fast-path for the common case when we're indexing 
    # by all scalars to retrieve a scalar result 
    if syntax_helpers.all_scalars(indices):
      data_ptr = self.attr(arr, "data")
      stride0 = self.strides(arr, 0)
      idx0 = self.assign_temp(self.tuple_proj(idx, 0), "idx_0" % i)
      offset_elts = self.mul(stride0, idx0, "total_offset")
      for i in xrange(1, n_required):
        idx_i = self.assign_temp(self.tuple_proj(idx, i), "idx_%d" % i)
        stride = self.strides(arr, i)
        elts_i = self.mul(stride, idx_i, "offset_elts_%d" % i)
        offset_elts = self.add(offset_elts, elts_i, "total_offset")
      return self.index(data_ptr, offset_elts, temp = False)
    else:
      return self.array_slice(arr, indices)
      
  def transform_Index(self, expr):
    return self.assign_temp(self.transform_lhs_Index(expr),"idx_result")
   
def lower_indexing(fn):
  return transform.cached_apply(LowerIndexing, fn)
