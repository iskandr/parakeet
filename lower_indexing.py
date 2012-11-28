import transform
 
import array_type 
import core_types 
import tuple_type
import syntax 
import syntax_helpers
class LowerIndexing(transform.Transform):
      
  def array_slice(self, arr, indices):
    print indices 
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
        if isinstance(start.type, core_types.NoneT):
          start = syntax_helpers.zero_i64
        stop = self.attr(idx, "stop")
        if isinstance(stop.type, core_types.NoneT):
          stop = shape_i 
        step = self.attr(idx, "step")
        if isinstance(step.type, core_types.NoneT):
          step = syntax_helpers.one_i64
              
        offset_i = self.mul(start, stride_i, "offset_%d" % i)
        elt_offset = self.add(elt_offset, offset_i)
        dim_i = self.sub(stop, start, "dim_%d" % i)
        # don't forget to cast shape elements to int64 
        new_shape.append(self.cast(dim_i, core_types.Int64))
        new_strides.append(self.mul(stride_i, step))
      else:
        raise RuntimeError("Unsupported index type: %s" % idx_t)

    elt_t = arr.type.elt_type 
    
    elt_size = syntax_helpers.const_int(elt_t.nbytes, core_types.Int64)
    byte_offset = self.mul(elt_offset, elt_size, "byte_offset") 
    new_data_ptr = self.incr_ptr(data_ptr, byte_offset)
    new_rank = len(new_strides)
    new_array_t = array_type.make_array_type(elt_t, new_rank)
    new_strides = self.tuple(new_strides, "strides")
    new_shape = self.tuple(new_shape, "shape")
    return syntax.ArrayView(new_data_ptr, new_shape, new_strides, type = new_array_t)
  
      
  def transform_lhs_Index(self, expr):
    arr = self.transform_expr(expr.value)
    arr_t = arr.type
    assert isinstance(arr_t, array_type.ArrayT), "Unexpected array %s : %s" % (arr, arr.type) 
    idx = self.assign_temp(self.transform_expr(expr.index), "idx")
    
    if self.is_tuple(idx):
      idx_t = idx.type
      n_elts = len(idx_t.elt_types)
      indices = [self.assign_temp(self.tuple_proj(idx, i), "idx_%d" % i) 
                 for i in xrange(n_elts)]
    else:
      indices = [idx]
    # right pad the index expression with None for each missing index
    n_given = len(indices)
    n_required = arr_t.rank 
    if n_given < n_required:
      extra_indices = [syntax_helpers.slice_none] * (n_required - n_given)
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

      result = self.array_slice(arr, indices)
      return result
      
  def transform_Assign(self, stmt):

    lhs = self.transform_lhs(stmt.lhs)
    rhs = self.transform_expr(stmt.rhs)
    assert not isinstance(lhs, syntax.Tuple), \
      "Too late in the compilation process to have tuples on the LHS"
    if isinstance(lhs.type, array_type.ArrayT) and isinstance(lhs, syntax.ArrayView):
      print "ARRAY LHS", lhs 
    return syntax.Assign(lhs, rhs)
  
  
  def transform_Index(self, expr):
    return self.assign_temp(self.transform_lhs_Index(expr),"idx_result")
   
def lower_indexing(fn):
  return transform.cached_apply(LowerIndexing, fn)
