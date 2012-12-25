import syntax 
from syntax import Assign 
from transform import Transform  
from collect_vars import collect_bindings 

def simple_block(block):
  return all(stmt.__class__ is Assign for stmt in block)


class LoopUnrolling():  
  def __init__(self, fn, unroll_factor = 4):
    Transform.__init__(self, fn)
    self.unroll_factor = unroll_factor 
    self.bindings = collect_bindings(self.fn)

  def find_loop_bounds(self, cond, body):
    pass 
      
  def transform_While(self, stmt):
    if simple_block(stmt.body):     
      start, stop, step = self.find_loop_bounds(stmt)
      if start is not None and stop is not None and step is not None:
        
    return stmt 
    
    
    