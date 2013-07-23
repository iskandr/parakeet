from ..syntax import Index, Map, unwrap_constant, zero_i64 
from transform import Transform

class LowerAdverbs(Transform):
    
  def transform_TypedFn(self, expr):
    from pipeline import loopify  
    return loopify(expr)
  
  def transform_ParFor(self, stmt):
    fn = self.transform_expr(stmt.fn)
    self.nested_loops(stmt.bounds, fn)
  
  def transform_IndexMap(self, expr, output = None):  
    # recursively descend down the function bodies to pull together nested ParFors
    fn = self.transform_expr(expr.fn)
    shape = expr.shape 
    
    dims = self.tuple_elts(shape)
    if len(dims) == 1:
      shape = dims[0]
    
    if output is None:
      output = self.create_output_array(fn, [shape], shape)

    def loop_body(idx):
      elt_result =  self.call(fn, (idx,))
      self.setidx(output, idx, elt_result)
    self.nested_loops(dims, loop_body)    
    return output
  
  def transform_IndexReduce(self, expr):
    init = self.transform_if_expr(self.init)
    fn = self.transform_expr(expr.fn)
    combine = self.transform_expr(expr.combine)
    shape = expr.shape 
    dims = self.tuple_elts(shape)
    n_loops = len(dims)
    
    if init is not None or not self.is_none(init):
      if n_loops > 1:
        zeros = self.tuple([zero_i64 for _ in xrange(n_loops)])
        init = self.call(fn, [zeros])
      else:
        init = self.call(fn, [zero_i64])
          
    def build_loops(index_vars, acc):    
      n_indices = len(index_vars)
      if n_indices > 0:
        acc_value = acc.get()
      else:
        acc_value = acc 
        
      if n_indices == n_loops:
        idx_tuple = self.tuple(index_vars) if n_indices > 1 else index_vars[0] 
        elt_result =  self.call(fn, (idx_tuple,))
        acc.update(self.call(combine, (acc_value, elt_result)))
      return acc.get()
        
      def loop_body(acc, idx):
        new_value = build_loops(index_vars + (idx,), acc = acc)
        acc.update(new_value)
        return new_value
      return self.accumulate_loop(self.int(0), dims[n_indices], loop_body, acc_value)
    return build_loops(index_vars = (), acc = init)

  def transform_IndexScan(self, expr):
    assert False, "IndexScan not implemented" 
  
  def transform_IndexFilter(self, expr):
    assert False, "IndexFilter not implemented"

  
  def transform_IndexFilterReduce(self, expr):
    assert False, "IndexFilterReduce not implemented"
    
  

  
