from ..syntax import Index, Map, unwrap_constant 
from transform import Transform

class LowerAdverbs(Transform):
  
  def transform_TypedFn(self, expr):
    from pipeline import loopify  
    return loopify(expr)
  
  def transform_adverb(self, expr, output=None):
    return expr.eval(self, output=output)
  
  def transform_IndexMap(self, expr, output = None):
    return self.transform_adverb(expr, output=output)
  
  def transform_IndexReduce(self, expr):
    return self.transform_adverb(expr)
    
    
  def transform_Map(self, expr, output = None):
    return self.transform_adverb(expr)
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    axis = unwrap_constant(expr.axis)
    return self.eval_map(fn, args, axis, output = output)

  def transform_Reduce(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    init = self.transform_if_expr(expr.init)
    axis = unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, init, args, axis)

  def transform_Scan(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    emit = self.transform_expr(expr.emit)
    init = self.transform_if_expr(expr.init)
    axis = unwrap_constant(expr.axis)
    return self.eval_scan(fn, combine, emit, init, args, axis)

  def transform_AllPairs(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    assert len(args) == 2
    x,y = self.transform_expr_list(args)
    axis = unwrap_constant(expr.axis)
    return self.eval_allpairs(fn, x, y, axis)
  
  def transform_Assign(self, stmt):
    if stmt.lhs.__class__ is Index and isinstance(stmt.rhs, Map):
      self.transform_Map(stmt.rhs, output = stmt.lhs)
      return None 
    return Transform.transform_Assign(self, stmt)
  
