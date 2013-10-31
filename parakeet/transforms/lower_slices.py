
from .. builder import build_fn 
from .. ndtypes import (NoneT, ScalarT, Int64, SliceT, TupleT, NoneType,repeat_tuple)
from .. syntax import  Index, Tuple, Var, ArrayView
from ..syntax.helpers import zero_i64, one_i64, all_scalars, slice_none, none


from transform import Transform
from phase import Phase 


class LowerSlices(Transform):  
  

  
  _setidx_cache = {}
  def make_setidx_fn(self, lhs_array_type, 
                             rhs_value_type, 
                             fixed_positions, 
                             slice_positions):
    fixed_positions = tuple(fixed_positions)
    slice_positions = tuple(slice_positions)
    key = lhs_array_type, rhs_value_type, fixed_positions, slice_positions
    if key in self._setidx_cache: 
      return self._setidx_cache[key]

    
    n_fixed_indices = len(fixed_positions)
    n_parfor_indices = len(slice_positions)
    n_indices = n_fixed_indices + n_parfor_indices
    parfor_idx_t = repeat_tuple(Int64, n_parfor_indices) if n_parfor_indices > 1 else Int64 
    fixed_index_types = [Int64] * n_fixed_indices
    slice_start_types = [Int64] * n_parfor_indices
    slice_step_types = [Int64] * n_parfor_indices
    input_types = [lhs_array_type, rhs_value_type] + fixed_index_types + \
      slice_start_types + slice_step_types + [parfor_idx_t]
    name = "setidx_array%d_%s_par%d" % \
      (lhs_array_type.rank, lhs_array_type.elt_type, n_parfor_indices)
      
    #build_fn
    # Inputs:  
    #  - input_types
    #  - return_type=NoneType
    #  - name=None
    #  - input_names=None
    # Outputs:
    #  f, builder, input_vars 
    idx_names = ["idx%d" % (i+1) for i in xrange(n_fixed_indices)]
    start_names = ["start%d" % (i+1) for i in xrange(n_parfor_indices)]
    step_names = ["step%d" % (i+1) for i in xrange(n_parfor_indices)]
    
    input_names = ["output_array", "input_array"] + idx_names + start_names + step_names + ["sliceidx"]
    
    fn, builder, input_vars = build_fn(input_types, NoneType, name, input_names)
    lhs = input_vars[0]
    rhs = input_vars[1]
    fixed_indices = input_vars[2:(2+n_fixed_indices)]
    starts = input_vars[(2+n_fixed_indices):(2+n_fixed_indices+n_parfor_indices)]
    steps = input_vars[(2+n_fixed_indices+n_parfor_indices):(2+n_fixed_indices+2*n_parfor_indices)]
    assert (2+n_fixed_indices+2*n_parfor_indices+1) == len(input_vars), \
      "Wrong number of vars: %s, expected %d but got %d" % \
      (input_vars, 2+n_fixed_indices+2*n_parfor_indices+1, len(input_vars))
    parfor_idx = input_vars[-1]
    if n_parfor_indices > 1:
      parfor_indices = builder.tuple_elts(parfor_idx)
    else:
      parfor_indices = [parfor_idx]
    
    indices = []   
    # interleave the fixed and slice indices in their appropriate order
    # This would be easier if I had OCaml cons lists!
    slice_counter = 0
    fixed_counter = 0
    for i in xrange(n_indices):
      if fixed_counter < len(fixed_positions)  and i == fixed_positions[fixed_counter]:
        indices.append(fixed_indices[fixed_counter])
        fixed_counter += 1 
      else:
        assert slice_counter < len(slice_positions)  and slice_positions[slice_counter] == i, \
          "Bad positions for indices, missing %d" % i  
        start = starts[slice_counter]
        step = steps[slice_counter]
        parfor_idx = parfor_indices[slice_counter]
        indices.append(builder.add(start, builder.mul(step, parfor_idx)))
        slice_counter += 1
    
    value = builder.index(rhs, parfor_indices)
    builder.setidx(lhs, builder.tuple(indices), value)
    builder.return_(none)
    self._setidx_cache[key] = fn
    return fn
    
    
  def dissect_index_expr(self, expr):
    """
    Split up an indexing expression into 
    fixed scalar indices and the start/stop/step of all slices
    """

    if isinstance(expr.index.type, TupleT):
      indices = self.tuple_elts(expr.index)
    else:
      indices = [expr.index]
    
    n_dims = expr.value.type.rank 
    n_indices = len(indices)
    assert n_dims >= n_indices, \
      "Not yet supported: more indices (%d) than dimensions (%d) in %s" % (n_indices, n_dims, expr) 
    if n_indices < n_dims:
      indices = indices + [slice_none] * (n_dims - n_indices)
    
    if all_scalars(indices):
      # if there aren't any slice expressions, don't bother with the rest of this function
      return indices, range(len(indices)), [], []
    
    shape = self.shape(expr.value)
    shape_elts = self.tuple_elts(shape)
    slices = []
    slice_positions = []
    scalar_indices = []
    scalar_index_positions = []
    
    for i, shape_elt in enumerate(shape_elts):
      idx = indices[i]
      t = idx.type
      if isinstance(t, ScalarT):
        scalar_indices.append(idx)
        scalar_index_positions.append(i)
      elif isinstance(t, NoneT):
        slices.append( (zero_i64, shape_elt, one_i64) )
        slice_positions.append(i)
      else:
        assert isinstance(t, SliceT), "Unexpected index type: %s in %s" % (t, expr) 
        start = zero_i64 if t.start_type == NoneType else self.attr(idx, 'start')
        stop = shape_elt if t.stop_type == NoneType else self.attr(idx, 'stop')
        step = one_i64 if t.step_type == NoneType else self.attr(idx, 'step')
        slices.append( (start, stop, step) )
        slice_positions.append(i)   
    return scalar_indices, scalar_index_positions, slices, slice_positions
    
  def transform_Index(self, expr):

    ndims = expr.value.type.rank
    if ndims == 1 and isinstance(expr.index.type, ScalarT):
      return expr
    elif isinstance(expr.index.type, TupleT) and \
        len(expr.index.type.elt_types) == ndims and \
        all(isinstance(t, ScalarT) for t in expr.index.type.elt_types):
      return expr 
    
    scalar_indices, scalar_index_positions, slices, slice_positions = \
      self.dissect_index_expr(expr)
    assert len(scalar_indices) == len(scalar_index_positions)
    assert len(slices) == len(slice_positions)
    if len(slices) == 0:
      return expr
    
    array = self.transform_expr(expr.value)
    data = self.attr(array, 'data')
    shape = self.shape(array)
    shape_elts = self.tuple_elts(shape)
    strides = self.strides(array)
     
    stride_elts = self.tuple_elts(strides)
    offset = self.attr(array, 'offset')
    new_shape = []
    new_strides = []
    n_indices = len(scalar_indices) + len(slices)
    fixed_count = 0
    slice_count = 0

    for i in xrange(n_indices):
      if fixed_count < len(scalar_index_positions) and scalar_index_positions[fixed_count] == i:
        idx = scalar_indices[fixed_count]
        fixed_count += 1
        offset = self.add(offset, self.mul(idx, stride_elts[i]), name = "offset")

      else:
        assert slice_positions[slice_count] == i
        (start, stop, step) = slices[slice_count]
        slice_count += 1
        dim_offset = self.mul(start, stride_elts[i], name = "dim_offset")
        offset = self.add(offset, dim_offset, "offset")
        span = self.sub(stop, start, name = "span")
        shape_elt = self.div_round_up(span, step, name = "new_shape")
        new_shape.append(shape_elt)
        stride_elt = self.mul(stride_elts[i], step, name = "new_stride")
        new_strides.append(stride_elt)
 
    size = self.prod(new_shape)
    new_rank = len(slices)
    assert len(new_shape) == new_rank
    assert len(new_strides) == new_rank

    new_array = self.array_view(data, self.tuple(new_shape), self.tuple(new_strides), offset, size)
    assert new_array.type.rank == new_rank
    #if len(stride_elts) > 1:
    #  print "STRIDES", stride_elts 
    #  print "EXPR", expr 
    #  print "NEW ARRAY", new_array
    #  print 
    return new_array
    
    
      
  def assign_index(self, lhs, rhs):
    if isinstance(lhs.index.type, ScalarT) and isinstance(rhs.type, ScalarT):
      self.assign(lhs,rhs)
      return 
    
    scalar_indices, scalar_index_positions, slices, slice_positions = \
      self.dissect_index_expr(lhs)
    assert len(scalar_indices) == len(scalar_index_positions)
    assert len(slices) == len(slice_positions)
    if len(slices) == 0:
      self.setidx(lhs.value, self.tuple(scalar_indices), rhs)
      return 
    

    # if we've gotten this far then there is a slice somewhere in the indexing 
    # expression, so we're going to turn the assignment into a setidx parfor 
    bounds = self.tuple([self.div(self.sub(stop, start), step)
                         for (start, stop, step) in slices])
    
    setidx_fn = self.make_setidx_fn(lhs.value.type, 
                                    rhs.type, 
                                    scalar_index_positions, 
                                    slice_positions)
    starts = [start for (start, _, _) in slices]
    steps = [step for (_, _, step) in slices]
    closure = self.closure(setidx_fn, 
                           [lhs.value, rhs] + scalar_indices + starts + steps)


    self.parfor(closure, bounds)
  
  def transform_TypedFn(self, expr):
    if self.fn.created_by is not None and isinstance (self.fn.created_by, Phase):
      return self.fn.created_by.apply(expr)
    import pipeline
    return pipeline.indexify.apply(expr)
  
  def transform_Assign(self, stmt):
    lhs_class = stmt.lhs.__class__
    rhs = self.transform_expr(stmt.rhs)
     
    if lhs_class is Tuple:
      for (i, _) in enumerate(stmt.lhs.type.elt_types):
        lhs_i = self.tuple_proj(stmt.lhs, i)
        rhs_i = self.tuple_proj(rhs, i)
        # TODO: make this recursive, otherwise nested
        # complex assignments won't get implemented
        assert lhs_i.__class__ not in (ArrayView, Tuple)
        if lhs_i.__class__ is Index:
          self.assign_index(lhs_i)
        else:
          assert lhs_i.__class__ is Var, "Unexpcted LHS %s : %s" % (lhs_i, lhs_i.type)
          self.assign(lhs_i, rhs_i)
      return None
    elif lhs_class is Index:
      
      self.assign_index(stmt.lhs, rhs)
      return None
    else:
      stmt.rhs = rhs
      return stmt
    


































