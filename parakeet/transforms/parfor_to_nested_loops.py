from transform import Transform

class ParForToNestedLoops(Transform):
  def transform_ParFor(self, stmt):
    fn = self.transform_expr(stmt.fn)
    self.nested_loops(stmt.bounds, fn)
    
    