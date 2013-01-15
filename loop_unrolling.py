import names 
import syntax_helpers 

from array_type import ArrayT
from clone_function import CloneFunction
from collect_vars import  collect_binding_names
from syntax import Assign, ForLoop, While, If, Return  
from syntax import Const, Var, Index, Tuple     
from syntax_helpers import const_int
from tuple_type import TupleT
from transform import Transform

def simple_assignment(lhs):
  if lhs.__class__ is Tuple:
    return all(simple_assignment(elt) for elt in lhs.elts)
  else:
    return lhs.__class__ is not Index or lhs.type.__class__ is not ArrayT

def simple_loop_body(stmts):
  for stmt in stmts:
    if stmt.__class__ in (Return, While, ForLoop):
      return False
    elif stmt.__class__ is If:
      if not simple_loop_body(stmt.true) or not simple_loop_body(stmt.false):
        return False
    elif stmt.__class__ is Assign and not simple_assignment(stmt.lhs):
      return False
  return True

def count_nested_stmts(stmt):
  if stmt.__class__ is If:
    return count_stmts(stmt.true) + count_stmts(stmt.false)
  elif stmt.__class__ is ForLoop or stmt.__class__ is While:
    return count_stmts(stmt.body)
  else:
    return 0
  
def count_stmts(stmts):
  return len(stmts) + sum(count_nested_stmts(stmt) for stmt in stmts)
  

class CloneStmt(CloneFunction):
  def __init__(self, outer_type_env):
    Transform.__init__(self)
    self.recursive = False
    self.type_env = outer_type_env
    self.rename_dict = {}

  def rename(self, old_name):
    old_type = self.type_env[old_name]
    new_name = names.refresh(old_name)
    new_var = Var(new_name, old_type)
    self.rename_dict[old_name] = new_var
    self.type_env[new_name] = old_type
    return new_name

  def rename_var(self, old_var):
    new_name = names.refresh(old_var.name)
    new_var = Var(new_name, old_var.type)
    self.rename_dict[old_var.name] = new_var
    self.type_env[new_name] = old_var.type
    return new_var

  def transform_merge(self, merge):
    new_merge = {}
    for (old_name, (l,r)) in merge.iteritems():
      new_name = self.rename(old_name)
      new_left = self.transform_expr(l)
      new_right = self.transform_expr(r)
      new_merge[new_name] = (new_left, new_right)
    return new_merge

  def transform_merge_before_loop(self, merge):
    new_merge = {}
    for (old_name, (l,r)) in merge.iteritems():
      new_name = self.rename(old_name)

      new_left = self.transform_expr(l)
      new_merge[new_name] = (new_left, r)
    return new_merge

  def transform_merge_after_loop(self, merge):
    for (new_name, (new_left, old_right)) in merge.items():
      merge[new_name] = (new_left, self.transform_expr(old_right))
    return merge

  def transform_Assign(self, expr):
    for name in collect_binding_names(expr.lhs):
      self.rename(name)
    new_lhs = self.transform_expr(expr.lhs)
    new_rhs = self.transform_expr(expr.rhs)

    return Assign(new_lhs, new_rhs)

  def transform_Var(self, expr):
    return self.rename_dict.get(expr.name, expr)

  def transform_ForLoop(self, stmt):
    new_var = self.rename_var(stmt.var)

    merge = self.transform_merge_before_loop(stmt.merge)
    new_start = self.transform_expr(stmt.start)
    new_stop = self.transform_expr(stmt.stop)
    new_step = self.transform_expr(stmt.step)
    new_body = self.transform_block(stmt.body)
    merge = self.transform_merge_after_loop(merge)
    return ForLoop(new_var, new_start, new_stop, new_step, new_body, merge)

def safediv(m,n):
  return (m+n-1)/n

class LoopUnrolling(Transform):
  def __init__(self, unroll_factor = 4, max_static_unrolling = 8):
    Transform.__init__(self)
    self.unroll_factor = unroll_factor
    if max_static_unrolling is not None:
    # should we unroll static loops more than ones with unknown iters? 
      self.max_static_unrolling = max_static_unrolling
    else:
      self.max_static_unrolling = unroll_factor

  def copy_loop_body(self, stmt, outer_loop_var, iter_num, phi_values = {}):
    """
    Assume the current codegen block is the unrolled loop
    """
    cloner = CloneStmt(self.type_env)
    # make a fresh copy of the loop
    loop = cloner.transform_ForLoop(stmt)
    i = const_int(iter_num, loop.var.type)
    loop_var_value = self.add(outer_loop_var, self.mul(stmt.step, i)) 
    self.assign(loop.var, loop_var_value)
    
    output_values = {}
    for (old_name, input_value) in phi_values.iteritems():
      new_var = cloner.rename_dict[old_name]
      self.assign(new_var, input_value)
      output_values[old_name] = loop.merge[new_var.name][1]
    self.blocks.top().extend(loop.body)
    return output_values 
    
  def transform_ForLoop(self, stmt):
    assert self.unroll_factor > 0
    if self.unroll_factor == 1:
      return stmt

    if stmt.step.__class__ is Const:
      assert stmt.step.value > 0, "Downward loops not yet supported"
    stmt = Transform.transform_ForLoop(self, stmt)
    
    if not simple_loop_body(stmt.body) or len(stmt.body) > 50:
      return stmt 
    
    start, stop, step = stmt.start, stmt.stop, stmt.step

    # if loop has static bounds, fully unroll unless it's too big
    unroll_factor = self.unroll_factor 
    
    if start.__class__ is Const and \
       stop.__class__ is Const and \
       step.__class__ is Const:
      niters = safediv(stop.value - start.value, step.value)
      if niters <= self.max_static_unrolling:
        unroll_factor = niters 
    
    # push the unrolled body onto the stack 
    self.blocks.push()
    phi_values = {}
    for (name, (input_value,_)) in stmt.merge.iteritems():
      phi_values[name] = input_value
    
    loop_var = self.fresh_var(stmt.var.type,  "i")
    for iter_num in xrange(self.unroll_factor):   
      phi_values = self.copy_loop_body(stmt, loop_var, iter_num, phi_values)            
    
    unrolled_body = self.blocks.pop()
    unroll_value = syntax_helpers.const_int(unroll_factor, stmt.var.type)
    unrolled_step = self.mul(unroll_value, stmt.step)
    trunc = self.mul(self.div(self.sub(stop,  start), unrolled_step), unrolled_step)
    
    unrolled_start = stmt.start
    unrolled_stop = self.add(stmt.start, trunc, "stop")
    
    final_merge = {}
    for (name, (input_value, _)) in stmt.merge.iteritems():
      output_value = phi_values[name]
      final_merge[name] = (input_value, output_value)
    
    unrolled_loop = ForLoop(var = loop_var,
                            start = unrolled_start,
                            stop = unrolled_stop,
                            step = unrolled_step,
                            body = unrolled_body,
                            merge = final_merge)
    

    
    if unrolled_stop.__class__ is Const and \
       stop.__class__ is Const and \
       unrolled_stop.value == stop.value:
      return unrolled_loop 
    else:
      self.blocks.append(unrolled_loop)
      # some loop iterations never got finished! 
      
      cleanup_loop_var = self.fresh_var(stmt.var.type, "cleanup_loop_counter")
      self.blocks.push()
      cleanup_phi_values = \
        self.copy_loop_body(stmt, cleanup_loop_var, 0, phi_values)
      cleanup_body = self.blocks.pop()
      
      cleanup_merge = {}
      for (name, input_value) in phi_values.iteritems():
        output_value = cleanup_phi_values[name]
        cleanup_merge[name] = (input_value, output_value)
      return ForLoop(var = cleanup_loop_var,
                     start = unrolled_loop.stop, 
                     stop = stmt.stop,
                     step = stmt.step, 
                     body = cleanup_body, 
                     merge = cleanup_merge)
    