from .. import prims 
from .. analysis.offset_analysis import OffsetAnalysis
from .. syntax import Var
from .. syntax.helpers import true, false 
from transform import Transform 

class OffsetPropagation(Transform):
  def pre_apply(self, fn):
    # map variables to pairs which include low/exclude high  
    self.known_ranges = {}
    self.known_offsets = OffsetAnalysis().visit_fn(fn)
  
  def transform_ForLoop(self, stmt):
    self.known_ranges[stmt.var.name] = (stmt.start, stmt.stop)
    stmt.body = self.transform_block(stmt.body)
    return stmt 
  
  def compare_lt(self, x, y, inclusive = True):
    if x.__class__ is Var:
      offsets = [(x.name, 0)]
      if x.name in self.known_offsets:
        offsets = self.known_offsets[x.name].union(set(offsets))
      for (var_name, offset) in offsets:
        if var_name in self.known_ranges:
          (low,high) = self.known_ranges[var_name]
          if y == high:
            if (offset >= 0 and inclusive) or (offset > 0):
              return false
          elif y == low:
            if (offset <= 0 and inclusive) or (offset < 0):
              return true
    return None 
  
  def transform_PrimCall(self, expr):
    p = expr.prim
    
    result = None 
    if isinstance(p, prims.Cmp):
      x,y = expr.args   
      if p == prims.less_equal:
        result = self.compare_lt(x,y, inclusive=True)
      elif p == prims.less:
        result = self.compare_lt(x,y, inclusive=False)
      elif p == prims.greater_equal:
        result = self.compare_lt(y, x, inclusive=True)
      elif p == prims.greater:
        result = self.compare_lt(y, x, inclusive=False)
      elif p == prims.equal:
        result1 = self.compare_lt(x, y, inclusive=True)
        result2 = self.compare_lt(y, x, inclusive=True)
        if result1 is not None and result2 is not None and result1 == result2:
          result = result1 
    if result is not None:
      return result
    else:
      return expr
      
