from ..analysis.value_range_analysis import (ValueRangeAnalyis, Interval, NoneValue)
from ..syntax.helpers import true, false 
from transform import Transform

class RangeTransform(Transform, ValueRangeAnalyis):
  """
  Base class for transforms which use value ranges gathered from analysis
  """
  
  def pre_apply(self, old_fn):
    ValueRangeAnalyis.__init__(self)
    self.visit_fn(old_fn)
    
  def cmp_eq(self, x, y):
    if not isinstance(x, Interval) or not isinstance(y, Interval):
      return None 
    if x.lower == y.lower and x.upper == y.upper:
      return true 
    if x.upper < y.lower or y.upper < x.lower:
      return false
    return None   
  
  def cmp_lt(self, x, y):
    if not isinstance(x, Interval) or not isinstance(y, Interval):
      return None 
    if x.upper < y.lower:
      return true 
    elif x.lower > y.upper:
      return false
    else:
      return None 
  
  def cmp_lte(self, x, y):
    if not isinstance(x, Interval) or not isinstance(y, Interval):
      return None 
    if x.upper < y.lower:
      return true 
    elif x.lower > y.upper:
      return false
    else:
      return None 
    