
from ..analysis.value_range_analysis import TupleOfIntervals, SliceOfIntervals
from ..ndtypes import ArrayT, TupleT, SliceT 
from ..syntax import Tuple, Slice, Var, Index  
from ..syntax.helpers import zero_i64, one_i64 

from range_transform import RangeTransform, Interval

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
    

    range_value = self.get(expr.index)

    if self.has_negative(range_value):
      if isinstance(range_value, TupleOfIntervals):
        index_ranges = range_value.elts 
        index_elts = list(self.tuple_elts(expr.index))
      else:
        index_ranges = [range_value]
        index_elts = [expr.index]
      assert len(index_ranges) == len(index_elts)
      shape = self.shape(expr.value)
      shape_elts = self.tuple_elts(shape)
      assert len(shape_elts) == len(index_elts), \
        "Mismatch between number of indices %d and rank %d in %s"  % \
        (len(index_elts), len(shape_elts), expr)
      for i, (index_range, index_elt) in enumerate(zip(index_ranges, index_elts)):
        if isinstance(index_range, Interval):
          if self.always_negative(index_range, inclusive=False):
            index_elts[i] = self.add(shape_elts[i], index_elt)
        elif isinstance(index_range, SliceOfIntervals) and self.has_negative(index_range):
          start = self.attr(index_elt, 'start', temp = False)
          stop = self.attr(index_elt, 'stop', temp = False)
          step = self.attr(index_elt, 'step', temp = False)
          shape_elt = shape_elts[i]
          
          if self.always_negative(index_range.start, inclusive=False):
            start = self.add(shape_elt, start)
          
          if self.always_negative(index_range.stop, inclusive=False):
            stop = self.add(shape_elt, stop)
          
          
          
          if self.always_negative(index_range.step):
            # if it's a negative step then (stupidly) there's
            # no valid stop value
            if self.is_none(start):
              start = self.sub(shape_elt, one_i64)
          else:
            # step is positive 
            if self.is_none(start):
              start = zero_i64
            if self.is_none(stop):
              stop = shape_elt
          index_elts[i] = Slice(start=start, stop=stop, step = step, type = index_elt.type)
      if len(index_elts) > 1:
        index = self.tuple(index_elts)
      else:
        index = index_elts[0]
      expr.index = index  
    return expr 
          
    
    
    
        