import names 
import syntax 
import syntax_helpers 

import clone_function
from clone_function import CloneFunction
from collect_vars import collect_bindings, collect_binding_names 
from nested_blocks import NestedBlocks
from subst import subst_stmt_list
from syntax import Assign, ForLoop, While, If, Return  
from syntax import Const, Var, Tuple     
from transform import Transform  


def simple_loop_body(stmts):
  for stmt in stmts:
    if stmt.__class__ in (Return, While, ForLoop):
      return False 
    elif stmt.__class__ is If:
      if not simple_loop_body(stmt.true) or not simple_loop_body(stmt.false):
        return False 
  return True 

class CloneStmt(CloneFunction):  
  def __init__(self, outer_type_env):
    Transform.__init__(self)
    self.recursive = False 
    self.type_env = outer_type_env
    self.rename_dict = {}
    
  def rename(self, old_name):
    old_type = self.type_env[old_name]
    new_name = names.refresh(old_name)
    self.type_env[new_name] = old_type
    new_var = Var(new_name, old_type)
    self.rename_dict[old_name] = new_var
    return new_name  
  
  def rename_var(self, old_var):
    new_name = names.refresh(old_var.name)
    self.type_env[new_name] = old_var.type 
    new_var = Var(new_name, old_var.type)
    self.rename_dict[old_var.name] = new_var 
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
    
  
  

class LoopUnrolling(Transform):  
  def __init__(self, unroll_factor = 4):
    Transform.__init__(self)
    self.unroll_factor = unroll_factor 


  def clone_loop(self, stmt):
    return CloneStmt(self.type_env).transform_ForLoop(stmt)
    

  def transform_ForLoop(self, stmt):
    assert stmt.step.__class__ is Const and stmt.step.value > 0, \
        "Downward loops not yet supported"
    if not simple_loop_body(stmt.body):
      return stmt
    if len(stmt.merge) > 0:
      print "Skipping simple loop since unrolling of phi-nodes not yet implemented"
      return stmt 
    counter_type = stmt.var.type
    unroll_value = syntax_helpers.const_int(self.unroll_factor, counter_type)
    
    iter_range = self.sub(stmt.stop,  stmt.start)
    trunc = self.div(iter_range, unroll_value)
    loop = self.clone_loop(stmt)
    loop_var = loop.var 
    loop_body = [] 
    loop_body.extend(loop.body) 
    loop.stop = self.add(stmt.start, trunc, "stop")
    loop.step = self.mul(loop.step, unroll_value)
    
    for i in xrange(1, self.unroll_factor):
      fresh_loop = self.clone_loop(loop)
      iter_num = self.add(loop_var, syntax_helpers.const_int(i, loop.var.type))
      loop_body.append(Assign(fresh_loop.var, iter_num))
      loop_body.extend(fresh_loop.body)
    loop.body = loop_body 
    self.blocks.append(loop)
    stmt.start = loop.stop 
    return stmt 
    