
from .. import syntax
from .. builder import Builder
 
from .. ndtypes import (NoneT, ScalarT, Int64, PtrT, IntT, 
                        SliceT, make_array_type, ArrayT, TupleT, NoneType,
                        repeat_tuple, make_tuple_type)
from .. syntax import Const, Index, Tuple, Var, ArrayView, Assign, Slice, Struct
from ..syntax.helpers import zero_i64, one_i64, all_scalars, slice_none 


from transform import Transform



class LowerSlices(Transform):
  
  def dissect_index_expr(self, expr):
    """
    Split up an indexing expression into 
    fixed scalar indices and the start/stop/step of all slices
    """
    if isinstance(expr.index.type, TupleT):
      indices = self.tuple_elts(expr.index)
    else:
      indices = [expr.index]
    
    n_dims = expr.array.rank 
    n_indices = len(indices)
    assert n_dims >= n_indices, \
      "Not yet supported: more indices (%d) than dimensions (%d) in %s" % (n_indices, n_dims, expr) 
    if n_indices < n_dims:
      indices = indices + [slice_none] * (n_dims - n_indices)
    
    if all_scalars(indices):
      # if there aren't any slice expressions, don't bother with the rest of this function
      return indices, range(len(indices)), [], []
    
    shape = self.shape(expr.array)
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
    
    
  def assign_index(self, lhs, rhs):
    if isinstance(lhs.index.type, ScalarT):
      self.assign(lhs,rhs)
      return 
    
    scalar_indices, scalar_index_positions, slices, slice_positions = \
      self.dissect_index_expr(lhs)
    assert len(scalar_indices) == len(scalar_index_positions)
    assert len(slices) == len(slice_positions)
    if len(slices) == 0:
      self.setidx(lhs.array, self.tuple(scalar_indices), rhs)
      return 
    # if we've gotten this far then there is a slice somewhere in the indexing 
    # expression, so we're going to turn the assignment into a setidx parfor 
    bounds = self.tuple([self.div(self.sub(stop, start), step)
                         for (start, stop, step) in slices])
    
    setidx_fn = None 
    self.parfor(setidx_fn, bounds)

      
  def transform_Assign(self, stmt):
    
    lhs_class = stmt.lhs.__class__
     
    if lhs_class is Tuple:
      for (i, _) in enumerate(stmt.lhs.type.elt_types):
        lhs_i = self.tuple_proj(stmt.lhs, i)
        rhs_i = self.tuple_proj(stmt.rhs, i)
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
      assert False
      self.assign_index(stmt.lhs, stmt.rhs)
      return None
    else:
      return stmt
    


































