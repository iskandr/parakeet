
from .. import syntax
from .. builder import Builder
from .. ndtypes import SliceT, make_array_type, ArrayT 
from .. ndtypes import NoneT, ScalarT, Int64, PtrT, IntT
from .. syntax import Const, Index, Tuple, Var, ArrayView, Assign, Slice, Struct
from ..syntax.helpers import zero_i64, one_i64

from transform import Transform

class LowerIndexing(Transform):
  def pre_apply(self, fn):
    self.bindings = {}

  def tuple_proj(self, tup, idx, explicit_struct = False):
    if tup.__class__ is Var and tup.name in self.bindings:
      stored = self.bindings[tup.name]
      if stored.__class__ is Tuple:
        return stored.elts[idx]
      else:
        return stored.args[idx]
    else:
      return Builder.tuple_proj(self, tup, idx,
                                  explicit_struct = explicit_struct)

  def attr(self, obj, field):
    if obj.__class__ is Var and obj.name in self.bindings:
      stored = self.bindings[obj.name]
      stored_class = stored.__class__
      if stored_class is Struct:
        pos = stored.type.field_pos(field)
        return stored.args[pos]
      elif stored_class  is Slice or stored_class is ArrayView:
        return getattr(stored, field)

    return Builder.attr(self, obj, field)


  def array_slice(self, arr, indices):
    data_ptr = self.attr(arr, "data")
    shape = self.attr(arr, "shape")
    strides = self.attr(arr, "strides")
    elt_offset = self.attr(arr, "offset")
    size = self.attr(arr, "size")

    new_strides = []
    new_shape = []

    for (i, idx) in enumerate(indices):
      stride_i = self.tuple_proj(strides, i)
      shape_i = self.tuple_proj(shape, i)
      idx_t = idx.type
      if isinstance(idx_t, ScalarT):
        offset_i = self.mul(idx, stride_i, "offset_%d" % i)
        elt_offset = self.add(elt_offset, offset_i)
      elif idx_t.__class__ is NoneT:
        new_strides.append(stride_i)
        new_shape.append(shape_i)
      elif idx_t.__class__ is SliceT:
        if isinstance(idx_t.start_type, NoneT):
          start =  zero_i64
        else:
          start = self.attr(idx, "start")
        
        if isinstance(idx_t.step_type, NoneT):
          step = one_i64
        else:
          step = self.attr(idx, "step")
        
        if isinstance(idx_t.stop_type, NoneT):
          stop = shape_i
        else:
          stop = self.attr(idx, "stop")
        
        offset_i = self.mul(start, stride_i, "offset_%d" % i)
        elt_offset = self.add(elt_offset, offset_i)
        dim_i = self.sub(stop, start, "dim_%d" % i)
        # don't forget to cast shape elements to int64
        new_shape.append(self.cast(dim_i, Int64))
        new_strides.append(self.mul(stride_i, step))
      else:
        raise RuntimeError("Unsupported index type: %s" % idx_t)

    elt_t = arr.type.elt_type
    new_rank = len(new_strides)
    new_array_t = make_array_type(elt_t, new_rank)
    new_strides = self.tuple(new_strides, "strides")
    new_shape = self.tuple(new_shape, "shape")
    return ArrayView(data_ptr, new_shape, new_strides,
                            elt_offset, size,
                            type = new_array_t)

  def transform_Index(self, expr):
    arr = self.transform_expr(expr.value)
    idx = self.transform_expr(expr.index)
    idx = self.assign_name(idx, "idx")

    arr_t = arr.type
    if arr_t.__class__ is PtrT:
      assert isinstance(idx.type, IntT)
      return expr

    assert arr_t.__class__ is  ArrayT, "Unexpected array %s : %s" % (arr, arr.type)

    if self.is_tuple(idx):
      indices = self.tuple_elts(idx)
    else:
      indices = [idx]
    # right pad the index expression with None for each missing index
    n_given = len(indices)
    n_required = arr_t.rank
    if n_given < n_required:
      extra_indices = [syntax.helpers.slice_none] * (n_required - n_given)
      indices.extend(extra_indices)

    # fast-path for the common case when we're indexing
    # by all scalars to retrieve a scalar result
    if syntax.helpers.all_scalars(indices):
      data_ptr = self.attr(arr, "data")
      strides = self.attr(arr, "strides")
      offset_elts = self.attr(arr, "offset")
      for (i, idx_i) in enumerate(indices):
        stride_i = self.tuple_proj(strides, i)

        elts_i = self.mul(stride_i, idx_i, "offset_elts_%d" % i)
        offset_elts = self.add(offset_elts, elts_i, "total_offset")
      return self.index(data_ptr, offset_elts, temp = False)
    else:
      return self.array_slice(arr, indices)

  def transform_Assign(self, stmt):
    lhs = stmt.lhs
    lhs_class = lhs.__class__
    rhs = self.transform_expr(stmt.rhs)
    if lhs_class is Tuple:
      for (i, _) in enumerate(lhs.type.elt_types):
        lhs_i = self.tuple_proj(lhs, i)
        rhs_i = self.tuple_proj(rhs, i)
        # TODO: make this recursive, otherwise nested
        # complex assignments won't get implemented
        assert lhs_i.__class__ not in (ArrayView, Tuple)
        self.assign(lhs_i, rhs_i)
      return None
    
    
    elif lhs_class is Index:
      lhs = self.transform_Index(lhs)
      if lhs.__class__ is ArrayView:
        copy_loop = self.array_copy(src = rhs, dest = lhs, return_stmt = True)
        copy_loop = self.transform_stmt(copy_loop)
        return copy_loop
    elif lhs_class is Var and \
         stmt.rhs.__class__ in (Slice, Struct, ArrayView, Tuple):
      self.bindings[lhs.name] = rhs

    return Assign(lhs, rhs)
