import syntax_helpers

from adverbs import Map 
from adverb_semantics import AdverbSemantics
from syntax import Index 
from transform import Transform

class LowerAdverbs(AdverbSemantics, Transform):
  def transform_TypedFn(self, expr):
    import pipeline 
    return pipeline.loopify(expr)
    
  def transform_Map(self, expr, output = None):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_map(fn, args, axis, output = output)

  def transform_Reduce(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    init = self.transform_if_expr(expr.init)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, init, args, axis)

  def transform_Scan(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    emit = self.transform_expr(expr.emit)
    init = self.transform_if_expr(expr.init)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_scan(fn, combine, emit, init, args, axis)

  def transform_AllPairs(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    assert len(args) == 2
    x,y = self.transform_expr_list(args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_allpairs(fn, x, y, axis)
  
  def transform_Assign(self, stmt):
    if stmt.lhs.__class__ is Index and isinstance(stmt.rhs, Map):
      self.transform_Map(stmt.rhs, output = stmt.lhs)
      return None 
    return Transform.transform_Assign(self, stmt)
  
