
from ..analysis.value_range_analysis import TupleOfIntervals, SliceOfIntervals
from ..ndtypes import ArrayT, TupleT, SliceT, ScalarT, NoneType 
from ..syntax import Tuple, Slice, Var, Index  
from ..syntax.helpers import zero_i64, one_i64, slice_none, make_slice_type, none

from range_transform import RangeTransform, Interval, NoneValue

class NegativeIndexElim(RangeTransform):
  
  """
  This doesn't entirely obviate the need for runtime checks on indexing 
  but does handle the simple case where a negative index can be inferred from the source 
  """
  
  def has_negative(self, range_value, inclusive = False):
    if isinstance(range_value, Interval):
      return self.always_negative(range_value, inclusive = inclusive)
    elif isinstance(range_value, TupleOfIntervals):
      return any(self.has_negative(elt) for elt in range_value.elts)
    elif isinstance(range_value, SliceOfIntervals):
      return self.has_negative(range_value.start) or \
        self.has_negative(range_value.stop) or \
        self.has_negative(range_value.step) 
    
  
  def transform_Index(self, expr):

    arr = expr.value 
    arr_t = arr.type 
    if arr_t.__class__ is not ArrayT:
      return expr 
    
    index = expr.index 
    
    if isinstance(index.type, (SliceT, ScalarT)):
      assert arr_t.rank > 0, "Unexpected zero-rank array in %s" % expr 
      extra_indices = [slice_none] * (arr_t.rank - 1) 
      index_elts = [index] + extra_indices
    elif isinstance(index.type, TupleT):
      n_given = len(index.type.elt_types)
      n_expected = arr_t.rank  
      assert n_given <= n_expected, "Too many indices in %s" % expr 
      
      # replace None in the indexing with Slice(None,None,None)
      given_elts = [given if given.type != NoneType else slice_none 
                    for given in self.tuple_elts(index)]
      index_elts = given_elts + [slice_none] * (n_expected - n_given)
      
    index_elts = list(index_elts)
    index_ranges = [self.get(idx_elt) for idx_elt in index_elts]
    
    assert len(index_ranges) == len(index_elts)
    shape = self.shape(expr.value, temp = False)
    shape_elts = self.tuple_elts(shape)
  
    assert len(shape_elts) == len(index_elts), \
      "Mismatch between number of indices %d and rank %d in %s"  % \
      (len(index_elts), len(shape_elts), expr)
    for i, (index_range, index_elt) in enumerate(zip(index_ranges, index_elts)):
      if isinstance(index_range, Interval):
        if index_range.upper == index_range.lower:
          index_elt = self.int(index_range.upper, "const_idx")
          
        if self.always_negative(index_range, inclusive=False):
          index_elts[i] = self.add(shape_elts[i], index_elt)
         
      elif isinstance(index_range, SliceOfIntervals) and self.has_negative(index_range):
        
        if isinstance(index_range.start, NoneValue):
          start = none
        elif index_range.start.upper == index_range.start.lower:
          start = self.int(index_range.start.lower)
        else:
          start = self.attr(index_elt, 'start', temp = False)
        
        if isinstance(index_range.stop, NoneValue):
          stop = none 
        elif index_range.stop.upper == index_range.stop.lower:
          stop = self.int(index_range.stop.lower)
        else:
          stop = self.attr(index_elt, 'stop', temp = False)
        
        if isinstance(index_range.step, NoneValue):
          step = none 
        elif index_range.step.lower == index_range.step.upper:
          step = self.int(index_range.step.lower)
        else: 
          step = self.attr(index_elt, 'step', temp = False)
          
        shape_elt = shape_elts[i]
        
        if self.always_negative(index_range.start, inclusive=False):
          start = self.add(shape_elt, start, "start")
        
        if self.always_negative(index_range.stop, inclusive=False):
          stop = self.add(shape_elt, stop, "stop")
          
        if self.always_negative(index_range.step):
          if self.is_none(start):
            start = self.sub(shape_elt, one_i64, "last_pos")
          
          #if self.is_none(stop):
          # if it's a negative step then (stupidly) there's
          # no valid stop value
          if self.is_none(stop):
            # from this point forward, negative stop just means include the element 0
            stop = self.int(-1)
            #assert False, \
            #  "Not yet supported: Python's peculiar behavior with negative slice steps and None stop"
          
            
        else:
          if self.is_none(step):
            step = one_i64 
            
          if self.is_none(start):
            start = zero_i64
            
          if self.is_none(stop):
            stop = shape_elt
        
        
        slice_t = make_slice_type(start.type, stop.type, step.type)
        index_elts[i] = Slice(start=start, stop=stop, step = step, type = slice_t)
      if len(index_elts) > 1:
        index = self.tuple(index_elts)
      else:
        index = index_elts[0]
      expr.index = index  

    return expr 
          
    
    
    
        