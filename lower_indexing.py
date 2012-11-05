import transform
 
import array_type 
import core_types 
import tuple_type
import syntax 
import syntax_helpers
class LowerIndexing(transform.Transform):
      
  def array_slice(self, arr, indices):
    data_ptr = self.attr(arr, "data") 
    shape = self.attr(arr, "shape")
    strides = self.attr(arr, "strides")
    
    new_strides = []
    new_shape = []
    elt_offset = syntax_helpers.zero_i64
    for (i, idx) in enumerate(indices):
      stride_i = self.tuple_proj(strides, i)
      shape_i = self.tuple_proj(shape, i)
      idx_t = idx.type 
      if isinstance(idx_t, core_types.ScalarT):
        offset_i = self.mul(idx, stride_i, "offset_%d" % i)
        elt_offset = self.add(elt_offset, offset_i)
      elif isinstance(idx_t, core_types.NoneT):
        new_strides.append(stride_i)
        new_shape.append(shape_i)
      elif isinstance(idx_t, array_type.SliceT):
        start = self.attr(idx, "start")
        stop = self.attr(idx, "stop")
        step = self.attr(idx, "step")
        offset_i = self.mul(start, stride_i, "offset_%d" % i)
        elt_offset = self.add(elt_offset, offset_i)
        dim_i = self.sub(stop, start, "dim_%d" % i)
        new_shape.append(dim_i)
        new_strides.append(self.mul(stride_i, step))
      else:
        raise RuntimeError("Unsupported index type: %s" % idx_t)

    elt_size = syntax_helpers.const_int(arr.type.nbytes(), core_types.Int64)
    byte_offset = self.mul(elt_offset, elt_size, "byte_offset") 
    new_data_ptr = self.incr_ptr(data_ptr, byte_offset)
    new_rank = len(new_strides)
    new_array_t = array_type.make_array_type(data_ptr.elt_type, new_rank)
    new_strides = self.tuple(new_strides, "strides")
    new_shape = self.tuple(new_shape, "shape")
    return syntax.ArrayView(new_data_ptr, new_shape, new_strides, type = new_array_t)
  
      
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
      strides = self.attr(arr, "strides")
      offset_elts = syntax_helpers.zero_i64
      for (i, idx_i) in enumerate(indices):
        stride_i = self.tuple_proj(strides, i)
        elts_i = self.mul(stride_i, idx_i, "offset_elts_%d" % i)
        offset_elts = self.add(offset_elts, elts_i, "total_offset")
      return self.index(data_ptr, offset_elts, temp = False)
    else:
      return self.array_slice(arr, indices)
      
  def transform_Index(self, expr):
    return self.assign_temp(self.transform_lhs_Index(expr),"idx_result")
   
def lower_indexing(fn):
  return transform.cached_apply(LowerIndexing, fn)
