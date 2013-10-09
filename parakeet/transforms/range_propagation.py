

from .. import prims 
from ..syntax import Const 
from ..analysis.value_range_analysis import Interval  
from range_transform import RangeTransform 

class RangePropagation(RangeTransform):
  
  def visit_Var(self, expr):
    if expr.name in self.ranges:
      range_val = self.ranges[expr.name]
      if isinstance(range_val, Interval) and range_val.lower == range_val.upper:
        return Const(range_val.lower, type = expr.type)
    return expr 
  
  def visit_PrimCall(self, expr):
    p = expr.prim 
    result = None 
    if isinstance(p, prims.Cmp):
      argx, argy = expr.args
      x, y = self.get(argx), self.get(argy)
      if x is not None and y is not None:        
        if p == prims.equal:
          result = self.cmp_eq(x,y)
        elif p == prims.less:
          result = self.cmp_lt(x,y)
        elif p == prims.less_equal:
          result = self.cmp_lte(x, y)
        elif p == prims.greater:
          result = self.cmp_lt(y, x)
        elif p == prims.greater_equal:
          result = self.cmp_lte(y, x)
     
          
    if result is None:
      return expr 
    else:
      return result 

  