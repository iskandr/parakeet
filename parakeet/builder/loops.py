
from .. import syntax 
from ..syntax import Expr, While, ForLoop 
from ..syntax.helpers import zero, one, zero_i32, zero_i64, wrap_if_constant
from core import BuilderCore 

class BuilderLoops(BuilderCore):
  
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