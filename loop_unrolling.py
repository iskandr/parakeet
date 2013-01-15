import syntax_helpers 

from array_type import ArrayT
from clone_stmt import CloneStmt
from syntax import Assign, ForLoop, While, If, Return, Var  
from syntax import Const,  Index, Tuple     
from syntax_helpers import const_int
from transform import Transform

def safediv(m,n):
  return (m+n-1)/n

class LoopUnrolling(Transform):
  def __init__(self, unroll_factor = 4, max_static_unrolling = 6):
    Transform.__init__(self)
    self.unroll_factor = unroll_factor
    if max_static_unrolling is not None:
    # should we unroll static loops more than ones with unknown iters? 
      self.max_static_unrolling = max_static_unrolling
    else:
      self.max_static_unrolling = unroll_factor

  def copy_loop_body(self, stmt, outer_loop_var, iter_num, phi_values = None):
    """Assume the current codegen block is the unrolled loop"""

    cloner = CloneStmt(self.type_env)
    # make a fresh copy of the loop
    loop = cloner.transform_ForLoop(stmt)
    i = const_int(iter_num, loop.var.type)
    loop_var_value = self.add(outer_loop_var, self.mul(stmt.step, i)) 
    self.assign(loop.var, loop_var_value)
    
    # if this isn't the first iteration of unrolling
    # then propagate old versions of phi-bound values
    # into this block
    if phi_values is not None:
      for (old_name, input_value) in phi_values.iteritems():
        new_var = cloner.rename_dict[old_name]
        self.assign(new_var, input_value)
        
    self.blocks.top().extend(loop.body)
    
    output_values = {}
    for old_name in stmt.merge.iterkeys():
        new_var = cloner.rename_dict[old_name]
        output_values[old_name] = loop.merge[new_var.name][1]
        
    return output_values, cloner.rename_dict
    
  def transform_ForLoop(self, stmt):
    assert self.unroll_factor > 0
    if self.unroll_factor == 1:
      return stmt

    if stmt.step.__class__ is Const:
      assert stmt.step.value > 0, "Downward loops not yet supported"
      
    stmt = Transform.transform_ForLoop(self, stmt)

    if not self.is_simple_block(stmt.body) or len(stmt.body) > 50:
      return stmt 
    
    start, stop, step = stmt.start, stmt.stop, stmt.step

    # if loop has static bounds, fully unroll unless it's too big
    unroll_factor = self.unroll_factor
    
    # number of iterations of loop iterations is not generally known  
    
    if start.__class__ is Const and \
       stop.__class__ is Const and \
       step.__class__ is Const:
      niters = safediv(stop.value - start.value, step.value)
      if niters <= self.max_static_unrolling:
        unroll_factor = niters 
    
    # push the unrolled body onto the stack
    self.blocks.push()
    
    phi_values = None    
    loop_var = self.fresh_var(stmt.var.type,  "i")
    name_mappings = None
    for iter_num in xrange(self.unroll_factor):   
      phi_values, curr_names = \
          self.copy_loop_body(stmt, loop_var, iter_num, phi_values) 

      if name_mappings is None:
        name_mappings = curr_names            
    
    unrolled_body = self.blocks.pop()
    unroll_value = syntax_helpers.const_int(unroll_factor, stmt.var.type)
    unrolled_step = self.mul(unroll_value, stmt.step)
    trunc = self.mul(self.div(self.sub(stop, start), unrolled_step),
                     unrolled_step)
    unrolled_stop = self.add(stmt.start, trunc)
       
    final_merge = {}
    for (old_name, (input_value, _)) in stmt.merge.iteritems():
      first_name_in_loop = name_mappings[old_name].name
      output_value = phi_values[old_name]
      final_merge[first_name_in_loop] = (input_value, output_value)
    
    unrolled_loop = ForLoop(var = loop_var,
                            start = stmt.start,
                            stop = unrolled_stop,
                            step = unrolled_step,
                            body = unrolled_body,
                            merge = final_merge)
    
    if unrolled_loop.start.__class__ is Const and \
       unrolled_loop.stop.__class__ is Const and \
       unrolled_loop.step.__class__ is Const:
      start_value = unrolled_loop.start.value
      stop_value = unrolled_stop.value
      step_value = unrolled_loop.step.value  
      if start_value + step_value == stop_value:
        self.assign(unrolled_loop.var, unrolled_loop.start)
        for (name, (input_value, _)) in final_merge.iteritems():
          var = Var(name, type = input_value)
          self.assign(var, input_value)
        self.blocks.top().extend(unrolled_body)
        return None 
    
    self.blocks.append(unrolled_loop)
    
    if unrolled_loop.stop.__class__ is not Const or \
       stop.__class__ is not Const or \
       unrolled_loop.stop.value != stop.value:
      cleanup_merge = {}
      for (old_name, (_, output_value)) in stmt.merge.iteritems():
        input_var = name_mappings[old_name]
        cleanup_merge[old_name] = (input_var, output_value)
      stmt.merge = cleanup_merge 
      stmt.start = unrolled_loop.stop 
      return stmt

