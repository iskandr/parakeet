from .. import prims 
from .. analysis.offset_analysis import OffsetAnalysis
from .. syntax import Var, Const 
from .. syntax.helpers import true, false, const
from transform import Transform 


class OffsetPropagation(Transform):
  def pre_apply(self, fn):
    # map variables to pairs which include low/exclude high  
    self.known_ranges = {}
    self.known_offsets = OffsetAnalysis().visit_fn(fn)
    self.seen_binding = set(fn.arg_names)

  def transform_Assign(self, stmt):
    result = Transform.transform_Assign(self, stmt )
    if result.lhs.__class__ is Var:
      self.seen_binding.add(result.lhs.name)
    return result 
  
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
  
  def const_additive_cancellation(self, x_name, y_value, output_type):
    """
    Return a single variable in situations like 
      x = prev_var + const 
      y = x + (- const) 
    """ 
    if x_name in self.known_offsets:
      for (prev_var_name, offset) in self.known_offsets[x_name]:
        if prev_var_name in self.seen_binding:
          if offset == -y_value:
            prev_type = self.fn.type_env[prev_var_name]
            prev_var = Var(name=prev_var_name, type = prev_type)
            return self.cast(prev_var, output_type)
  
  def var_subtractive_cancellation(self, x_name, y_name, output_type):
    """
    Return a single constant in cases like:
      a = b + const 
      c = b - a
    """
    if x_name in self.known_offsets:
      for (prev_var_name, offset) in self.known_offsets[x_name]:
        if prev_var_name == y_name:
          return self.cast(const(offset), output_type)
        
  
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
          
          
    # THIS IS MUCH MORE LIMITED THAN THE SORT OF SYMBOLIC REWRITES
    # WE COULD DO IN GENERAL, i.e. y = x + 2, z = x - 2, (y-z) ==> 4
    elif p == prims.add:
      x, y = expr.args
      result = None  
      if x.__class__ is Var and y.__class__ is Const:
        result = self.const_additive_cancellation(x.name, y.value, expr.type)
      elif x.__class__ is Const and y.__class__ is Var:
        result = self.const_additive_cancellation(y.name, x.value, expr.type)
        
      if result is not None: return result
         
    elif p == prims.subtract:

      x, y = expr.args 
      result = None 
      if x.__class__ is Var and y.__class__ is Const:
        result = self.const_additive_cancellation(x.name, -y.value, expr.type)
      elif x.__class__ is Var and y.__class__ is Var:
        result = self.var_subtractive_cancellation(x.name, y.name, expr.type)
      if result is not None: return result
    if result is not None:
      return result
    else:
      return expr
      
