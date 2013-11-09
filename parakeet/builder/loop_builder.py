
from .. import syntax 
from ..ndtypes import ScalarT, TupleT 
from ..syntax import Expr, While, ForLoop, Const 
from ..syntax.helpers import zero, one, zero_i32, zero_i64, wrap_if_constant

from array_builder import ArrayBuilder 

class LoopBuilder(ArrayBuilder):
  
  """
  Builder for loops and things that use loops
  """
  
  
  
  def loop_var(self, name = "i", start_val = syntax.zero_i64):
    """
    Generate three SSA variables to use as the before/during/after values
    of a loop counter throughout some loop.

    By default initialize the counter to zero, but optionally start at different
    values using the 'start_val' keyword.
    """

    start_val = wrap_if_constant(start_val)
    counter_type = start_val.type

    counter_before = self.assign_name(start_val, name + "_before")
    counter = self.fresh_var(counter_type, name)
    counter_after = self.fresh_var(counter_type, name + "_after")
    merge = {counter.name:(counter_before, counter_after)}
    return counter, counter_after, merge

  def loop(self, 
           start, 
           niters, 
           loop_body,
           return_stmt = False,
           while_loop = False,
           step = None, 
           merge = None):
    
    if isinstance(start, (int,long)):
      start = self.int(start)
      
    assert isinstance(start, Expr)
    
    if isinstance(niters, (int,long)):
      niters = self.int(niters)
      
    assert isinstance(niters, Expr)
    
    if step is None:
      step = one(start.type)
    elif isinstance(step, (int, long)):
      step = self.int(step)
    
    assert isinstance(step, Expr)
    
    if merge is None:
      merge = {}
      
    if while_loop:
      i, i_after, i_merge = self.loop_var("i", start)
      merge.update(i_merge)
      cond = self.lt(i, niters)
      self.blocks.push()
      loop_body(i)
      self.assign(i_after, self.add(i, step))
      body = self.blocks.pop()
      loop_stmt = While(cond, body, merge)
    else:
      var = self.fresh_var(start.type, "i")
      self.blocks.push()
      loop_body(var)
      body = self.blocks.pop()
      loop_stmt = ForLoop(var, start, niters, step, body, merge)

    if return_stmt:
      return loop_stmt
    else:
      self.blocks += loop_stmt


  class Accumulator:
    def __init__(self, acc_type, fresh_var, assign):
      self.acc_type = acc_type
      self.fresh_var = fresh_var
      self.assign = assign
      self.start_var = fresh_var(acc_type, "acc")
      self.curr_var = self.start_var

    def get(self):
      return self.curr_var

    def update(self, new_value):
      new_var = self.fresh_var(self.acc_type, "acc")
      self.assign(new_var, new_value)
      self.curr_var = new_var

  def mk_acc(self, init):
    return self.Accumulator(init.type, self.fresh_var, self.assign)
    
  def accumulate_loop(self, start, stop, loop_body, init, return_stmt = False):
    acc = self.mk_acc(init)
    def loop_body_with_acc(i):
      loop_body(acc, i)
    loop_stmt = self.loop(start, stop, loop_body_with_acc, return_stmt = True)
    loop_stmt.merge[acc.start_var.name] = (init, acc.curr_var)
    if return_stmt:
      return loop_stmt, acc.start_var
    else:
      self.blocks += loop_stmt
      return acc.start_var 
    
  def array_copy(self, src, dest, return_stmt = False):
    assert self.is_array(dest)
    shape = self.shape(dest)
    dims = self.tuple_elts(shape)
    rank = len(dims)
    index_vars = []
    def create_loops():
      i = len(index_vars)
      def loop_body(index_var):
        index_vars.append(index_var)
        if i+1 == rank:
          index_tuple = self.tuple(index_vars, "idx")
          lhs = self.index(dest, index_tuple, temp=False)
          rhs = self.index(src, index_tuple, temp=True)
          self.assign(lhs, rhs)
        else:
          self.blocks += create_loops()
      start = syntax.zero_i64
      stop = dims[i]
      if i > 0 or return_stmt:
        return self.loop(start, stop, loop_body, True)
      else:
        return self.loop(start, stop, loop_body, return_stmt)

    return create_loops()
  
  def _to_list(self, bounds):
    if isinstance(bounds, Expr):
      if isinstance(bounds.type, ScalarT):
        return [bounds]
      else:
        assert isinstance(bounds.type, TupleT), \
          "Expected tuple but got %s : %s" % (bounds, bounds.type) 
        return self.tuple_elts(bounds)
    elif isinstance(bounds, (int,long)):
      return [bounds]
    else:
      assert isinstance(bounds, (tuple,list))
      return bounds 
  
  
  def nested_loops(self, 
                     upper_bounds, 
                     loop_body, 
                     lower_bounds = None, 
                     step_sizes = None, 
                     index_vars_as_list = False):
    upper_bounds = self._to_list(upper_bounds)
    
    n_loops = len(upper_bounds)
    assert lower_bounds is None or len(lower_bounds) == n_loops
    assert step_sizes is None or len(step_sizes) == n_loops 
    
    if lower_bounds is None:
      lower_bounds = [self.int(0) for _ in upper_bounds]
    else:
      lower_bounds = self._to_list(lower_bounds)
    
    if step_sizes is None:
      step_sizes = [self.int(1) for _ in upper_bounds]
    else:
      step_sizes = self._to_list(step_sizes)
    
    
    def build_loops(index_vars = ()):
      n_indices = len(index_vars)
      if n_indices == n_loops:
        if isinstance(loop_body, Expr):
          input_types = self.input_types(loop_body)
          if len(input_types) == len(index_vars):
            result = self.call(loop_body, index_vars)
          else:
            result = self.call(loop_body, [self.tuple(index_vars)])
        else:
          if index_vars_as_list:
            idx_tuple = list(index_vars)
          elif n_indices > 1:
            idx_tuple = self.tuple(index_vars)
          else:
            idx_tuple = index_vars[0]
          assert hasattr(loop_body, '__call__'), "Expected callable value, got %s" % (loop_body,)
          result = loop_body(idx_tuple)
        assert self.is_none(result), "Expected loop body to return None, not %s" % (result,)
      else:
        def inner_loop_body(idx):
          build_loops(index_vars + (idx,))
        lower = lower_bounds[n_indices]
        upper = upper_bounds[n_indices]  
        step = step_sizes[n_indices]
        # sanity check the bounds 
        if lower.__class__ is Const and upper.__class__ is Const and step.__class__ is Const:
          if step.value < 0:
            assert lower.value > upper.value, \
              "Attempting to build invalid loop from %s to %s by %s" % (lower, upper, step)
          elif step.value > 0:
            assert upper.value > lower.value, \
              "Attempting to build invalid loop from %s to %s by %s" % (lower, upper, step) 
        self.loop(lower, upper, inner_loop_body, step=step)
    build_loops()