import prims 

from offset_analysis import OffsetAnalysis
from syntax import Var
from syntax_helpers import true, false 
from transform import Transform 

class RangePropagation(Transform):
  def pre_apply(self, fn):
    # map variables to pairs which include low/exclude high  
    self.known_ranges = {}
    
    self.known_offsets = OffsetAnalysis().visit_fn(fn)
  
  def transform_ForLoop(self, stmt):
    self.known_ranges[stmt.var.name] = (stmt.start, stmt.stop)
    stmt.body = self.transform_block(stmt.body)
    return stmt 
  
  def transform_PrimCall(self, expr):
    if expr.prim == prims.less_equal:
      x, y = expr.args
      
      if x.__class__ is Var:
        offsets = [(x.name, 0)]
        if x.name in self.known_offsets:
          offsets = self.known_offsets[x.name].union(set(offsets))
        for (var_name, offset) in offsets:
          if var_name in self.known_ranges:
            (low,high) = self.known_ranges[var_name]
            if y == high and offset >= 0:
              return false 
            elif y == low and offset <= 0:
              return true
          
    return expr
      
