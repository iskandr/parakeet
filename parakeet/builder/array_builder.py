
from ..ndtypes import (make_slice_type, make_array_type, ptr_type, 
                       ArrayT, TupleT, ScalarT, Type, PtrT, Int64, IntT, Float64)
from ..syntax import (Alloc, AllocArray, ArrayView, Const, Index, Slice, Struct, Var, Select, Expr)
from ..syntax.helpers import (const, zero_i64, wrap_if_constant, slice_none, unwrap_constant)
from arith_builder import ArithBuilder

class ArrayBuilder(ArithBuilder):
  """
  Builder for constructing arrays and getting their properties
  """
  
  


  def elt_type(self, x):
    if isinstance(x, Type):
      try:
        return x.elt_type
      except:
        return x
    elif self.is_array(x):
      return x.type.elt_type
    else:
      return x.type
  
  def alloc_array(self, elt_t, dims, name = "array", 
                   explicit_struct = False, 
                   array_view = False, 
                   order = "C"):
    """
    Given an element type and sequence of expressions denoting each dimension
    size, generate code to allocate an array and its shape/strides metadata. For
    now I'm assuming that all arrays are in row-major, eventually we should make
    the layout an option.
    """
    
    assert order == "C", "Only row-major layout supported so far, not %s" % order 

    if self.is_tuple(dims):
      shape = dims
      dims = self.tuple_elts(shape)
    else:
      if not isinstance(dims, (list, tuple)):
        dims = [dims]
      shape = self.tuple(dims, "shape", explicit_struct = explicit_struct)
    rank = len(dims)
    array_t = make_array_type(elt_t, rank)
    if explicit_struct or array_view:
      nelts = self.prod(dims, name = "nelts")
      ptr_t = ptr_type(elt_t)

      ptr_var = self.assign_name(Alloc(elt_t, nelts, type = ptr_t), "data_ptr")
      stride_elts = [const(1)]

      # assume row-major for now!

      for d in reversed(dims[1:]):
        next_stride = self.mul(stride_elts[0], d, "dim")
        stride_elts = [next_stride] + stride_elts
      strides = self.tuple(stride_elts, "strides", explicit_struct = explicit_struct)
      if explicit_struct:
        array = Struct([ptr_var, shape, strides, zero_i64, nelts], type = array_t)
      else:
        array = ArrayView(data = ptr_var, 
                          shape = shape, 
                          strides = strides,
                          offset = zero_i64,
                          size = nelts, 
                          type = array_t)
    else:
      array = AllocArray(shape, elt_type = elt_t, type = array_t)
    if name is None: 
      return array 
    return self.assign_name(array, name)
  
  def len(self, array):
    return self.shape(array, 0)
  
  def nelts(self, array, explicit_struct = False):
    shape_elts = self.tuple_elts(self.shape(array), explicit_struct = explicit_struct)
    return self.prod(shape_elts, name = "nelts") 

  def rank(self, value):
    if self.is_array(value):
      return value.type.rank
    else:
      return 0

  def shape(self, array, dim = None, explicit_struct = False, temp = True):
    if isinstance(array.type, ArrayT):
      shape = self.attr(array, "shape", temp = temp)
      if dim is None:
        return shape
      if isinstance(dim, Expr):
        dim = unwrap_constant(dim)
      assert isinstance(dim, (int, long)), "Expected array dimension to be an int, got %s" % dim 
      dim_value = self.tuple_proj(shape, dim, explicit_struct = explicit_struct)
      if temp:
        return self.assign_name(dim_value, "dim%d" % dim)
      else:
        return dim_value 
    else:
      return self.tuple([])

  def strides(self, array, dim = None, explicit_struct = False):
    assert array.type.__class__ is ArrayT
    strides = self.attr(array, "strides")
    if dim is None:
      return strides
    else:
      elt_value = self.tuple_proj(strides, dim, explicit_struct = explicit_struct)
      return self.assign_name(elt_value, "stride%d" % dim)

  def slice_value(self, start, stop, step):
    slice_t = make_slice_type(start.type, stop.type, step.type)
    return Slice(start, stop, step, type = slice_t)

  def build_slice_indices(self, rank, axis, idx):
    """
    Build index tuple to pull out the 'idx' element along the given axis
    """
    if rank == 1:
      assert axis == 0
      return idx
    
    indices = []
    for i in xrange(rank):
      if i == axis:
        indices.append(idx)
      else:
        s = self.slice_value(self.none, self.none, self.int(1))
        indices.append(s)
    return self.tuple(indices)

  def elts_in_slice(self, start, stop, step):
    start_minus_start = self.sub(stop, start, name = "stop_minus_start")
    nelts = self.div(self.cast(start_minus_start, Float64), step, name = "nelts")
    ceil = self.ceil(nelts)
    nelts = self.cast(ceil, Int64)
    return self.max(nelts, self.zero(nelts.type))

  def slice_along_axis(self, arr, axis, idx):
    """
    Pull out a slice if the array has the given axis, 
    otherwise just return the array 
    """
    r = self.rank(arr)
    
    if isinstance(axis, Expr):
      axis = unwrap_constant(axis)
    idx = wrap_if_constant(idx)
    
    if r == 1 and (axis is None or axis == 0):
      return self.index(arr, idx)
    elif axis is None:
      if isinstance(idx.type, ScalarT):
        idx = self.tuple((idx,) * r)
      return self.index(arr, idx)
    elif r > axis:
      index_tuple = self.build_slice_indices(r, axis, idx)
      return self.index(arr, index_tuple)
    else:
      return arr

  def output_slice(self, output, axis, idx):
    """
    Create an expression which acts as an LHS output location 
    for a slice throught the variable 'output' along the given axis
    """
    r = self.rank(output)
    if r > 1:
      output_indices = self.build_slice_indices(r, axis, idx)
    elif r == 1:
      output_idx = self.slice_value(idx, self.none, self.int(1))
      output_indices = self.tuple([output_idx])
    else:
      output_idx = self.slice_value(self.none, self.none, self.none)
      output_indices = self.tuple([output_idx])
    return self.index(output, output_indices)
  

  def size_along_axis(self, value, axis):
    return self.shape(value, axis)

  def check_equal_sizes(self, sizes):
    pass
  
  def index(self, arr, idx, temp = False, name = None):
    """Index into array or tuple differently depending on the type"""
    
    temp = temp or name is not None
    
    arr_t = arr.type

    if isinstance(arr_t, ScalarT):
      # even though it's not correct externally, it's
      # often more convenient to treat indexing
      # into scalars as the identity function.
      # Just be sure to catch this as an error in
      # the user's code earlier in the pipeline.
      return arr
    
    if isinstance(arr_t, TupleT):
      if isinstance(idx, Const):
        idx = idx.value

      assert isinstance(idx, int), \
          "Index into tuple must be an integer, got %s" % idx
      if isinstance(idx, Const):
        idx = idx.value
      proj = self.tuple_proj(arr, idx)
      if temp:
        return self.assign_name(proj, "tuple_elt%d" % idx if name is None else name)
      else:
        return proj

    if self.is_tuple(idx):
      indices = self.tuple_elts(idx)
    elif isinstance(idx, (list,tuple)) or hasattr(idx, '__iter__'):
      indices = tuple(map(wrap_if_constant,idx))
    else:
      indices = (wrap_if_constant(idx),)

    n_required = arr_t.rank
    n_indices = len(indices)
    if n_indices < n_required:
      # all unspecified dimensions are considered fully sliced
      extra = (slice_none,) * (n_required - n_indices)
      indices = indices + extra

    if len(indices) > 1:
      idx = self.tuple(indices, name = name)
    else:
      idx = indices[0]

    t = arr_t.index_type(idx.type)
    idx_expr = Index(arr, idx, type=t)
    if temp:
      return self.assign_name(idx_expr, "array_elt" if name is None else name)
    else:
      return idx_expr

  def index_along_axis(self, arr, axis, idx, name=None):
    if arr.type.__class__ is not ArrayT:
      return arr
    assert isinstance(axis, int), \
        "Axis must be a known constant int, got: " + str(axis)
    indices = []
    for i in xrange(arr.type.rank):
      if i == axis:
        indices.append(wrap_if_constant(idx))
      else:
        indices.append(slice_none)

    index_tuple = self.tuple(indices)


    result_t = arr.type.index_type(index_tuple.type)
    idx_expr = Index(arr, index_tuple, type=result_t)
    if name:
      return self.assign_name(idx_expr, name)
    else:
      return idx_expr

  def setidx(self, arr, idx, v):
    self.assign(self.index(arr, idx, temp=False), v)


  def array_view(self, data, shape, strides, offset, nelts):
    assert isinstance(data.type, PtrT), \
        "Data field of array must be a pointer, got %s" % data.type
    if data.__class__ is not Var:
      data = self.assign_name(data, "data_ptr")
      
    if isinstance(shape.type, ScalarT):
      shape = self.tuple([shape])
    assert isinstance(shape.type, TupleT), \
      "Shape of array must be a tuple, got: %s" % shape.type
      
    if isinstance(strides.type, ScalarT):
      strides = self.tuple(strides)
      
    assert isinstance(strides.type, TupleT), \
      "Strides of array must be a tuple, got: %s" % strides.type

    ndims = len(strides.type.elt_types)
    assert ndims == len(shape.type.elt_types)

    elt_t = data.type.elt_type
    array_t = ArrayT(elt_t, ndims)
    return ArrayView(data = data, shape = shape, strides = strides, 
                     offset = offset, size = nelts, 
                     type = array_t)

  
    