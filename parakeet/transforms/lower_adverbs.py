from ..ndtypes import TupleT, Int64 
from ..syntax import Index, Map, unwrap_constant, zero_i64 
from transform import Transform
from parakeet.syntax.stmt import ForLoop

class LowerAdverbs(Transform):
    
  def transform_TypedFn(self, expr):
    from pipeline import loopify  
    return loopify(expr)
  

  
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
  
  _loop_counters = ["i", "j", "k", "l", "ii", "jj", "kk", "ll"]
  
  def get_loop_counter(self, depth):
    loop_counter_name = self._loop_counters[depth % len(self._loop_counters)] 
    return  self.fresh_var(Int64, loop_counter_name)
  
   
  def build_nested_reduction(self, indices, starts, bounds, old_acc, body_fn):
      if len(bounds) == 0:
        if len(indices) > 1:
          indices = self.tuple(indices)
        else:
          indices = indices[0]
        new_acc = body_fn(indices, old_acc)
        return new_acc 
      else:
        acc_t = old_acc.type 
        acc_before = self.fresh_var(acc_t, "acc_before")
        loop_counter = self.get_loop_counter(len(indices)) 
        
        future_indices = indices + (loop_counter,)
        future_starts = starts[1:]
        future_bounds = bounds[1:]
        self.blocks.push()
        acc_after = self.build_nested_reduction(future_indices, 
                                                future_starts, 
                                                future_bounds, 
                                                acc_before, 
                                                body_fn)
        body = self.blocks.pop()
        merge = {acc_before.name : (old_acc, acc_after)}
        for_loop = ForLoop(var = loop_counter, 
                           start = starts[0],
                           stop = bounds[0], 
                           step = self.int(1), 
                           body = body, 
                           merge = merge)
        self.blocks.append_to_current(for_loop)
        return acc_before # should this be acc_after?

  
  def transform_IndexReduce(self, expr):
    init = self.transform_if_expr(expr.init)
    fn = self.transform_expr(expr.fn)
    combine = self.transform_expr(expr.combine)
    shape = expr.shape 
    # n_loops = len(dims)
    
    assert init is not None, "Can't have empty 'init' field for IndexReduce"
   
    if isinstance(shape.type, TupleT): 
      bounds = self.tuple_elts(shape)
    else:
      bounds = [shape]
      
    if expr.start_index is not None:
      if isinstance(expr.start_index.type, TupleT):
        starts = self.tuple_elts(expr.start_index)
      else:
        starts = [expr.start_index]
    else:
      starts = [self.int(0)] * len(bounds)
      
    
    def body(indices, old_acc):
      elt = self.call(fn, (indices,))
      return self.call(combine, (old_acc, elt))
       
    return self.build_nested_reduction(
              indices = (), 
              starts = starts, 
              bounds = bounds, 
              old_acc = init, 
              body_fn = body)


  def transform_IndexScan(self, expr, output = None):
    init = self.transform_if_expr(expr.init)
    fn = self.transform_expr(expr.fn)
    combine = self.transform_expr(expr.combine)
    
    shape = expr.shape 
    if output is None:
      output = self.create_output_array(fn, [shape], shape)
      
    
    assert init is not None, "Can't have empty 'init' field for IndexScan"
    assert init.type == self.return_type(fn), \
      "Mismatching types init=%s, fn returns %s" % (init.type, self.return_type(fn))
    if isinstance(shape.type, TupleT): 
      bounds = self.tuple_elts(shape)
    else:
      bounds = [shape]
      
    def body(indices, old_acc):
      elt = self.call(fn, (indices,))
      new_acc = self.call(combine, (old_acc, elt))
      self.setidx(output, indices, new_acc)
      return new_acc
    starts = [self.int(0)] * len(bounds)
    self.build_nested_reduction(indices = (),
                                starts = starts, 
                                bounds = bounds, 
                                old_acc = init, body_fn = body)
    return output 
  
  def transform_IndexFilter(self, expr):
    assert False, "IndexFilter not implemented"

  
  def transform_IndexFilterReduce(self, expr):
    assert False, "IndexFilterReduce not implemented"
    
  

  
