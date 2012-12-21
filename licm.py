import syntax 
from syntax import Var, Assign, Return, While, If, Tuple 
from syntax_visitor import SyntaxVisitor 
from scoped_set import ScopedSet
from collect_vars import collect_var_names, collect_binding_names 
from  tuple_type import TupleT  
from closure_type import ClosureT   
    
 
class Find_LICM_Candidates(SyntaxVisitor):
  def __init__(self):
    self.mutable_types = None
    self.volatile_vars = ScopedSet()
    self.depends_on = {}
    self.safe_to_move = set([])
    self.curr_block_id = None 
    self.block_contains_return = set([])
  
  def mark_safe_assignments(self, block, volatile_set):
    for stmt in block:
      klass = stmt.__class__ 
      if klass is Assign and \
          stmt.lhs.__class__ is Var:           
        name = stmt.lhs.name 
        dependencies = self.depends_on.get(name, set([]))
        volatile = name in volatile_set or any(d in volatile_set for d in dependencies)
        if not volatile:
          self.safe_to_move.add(name)
      # just in case there are Returns in nested control flow 
      # we should probably avoid changing the performance characteristics
      # by pulling out statements which will never run
      elif klass is If:
        if id(stmt.true) in self.block_contains_return or \
            id(stmt.false) in self.block_contains_return:
          break
      elif klass is While:
        if id(stmt.body) in self.block_contains_return:
          break 
      elif klass is Return:
        break 
      
  def mark_curr_block_returns(self):
    self.block_contains_return.add(self.curr_block_id)
  
  def does_block_return(self, block):
    return id(block) in self.block_contains_return
  
  def visit_Return(self, stmt):
    self.mark_curr_block_returns()
  
  def visit_block(self, stmts):
    self.curr_block_id = id(stmts) 
    SyntaxVisitor.visit_block(self, stmts)
      
  def visit_merge(self, merge, both_branches = True):
    pass 
          
  def visit_While(self, stmt):
    self.volatile_vars.push(stmt.merge.keys())
    SyntaxVisitor.visit_While(self, stmt)
    if self.does_block_return(stmt.body):
      self.block_contains_return()
    volatile_in_scope = self.volatile_vars.pop()
    self.mark_safe_assignments(stmt.body, volatile_in_scope)
    
  def visit_If(self, stmt):
    self.volatile_vars.push(stmt.merge.keys())
    SyntaxVisitor.visit_If(self, stmt)
    if self.does_block_return(stmt.true) or self.does_block_return(stmt.false):
      self.mark_curr_block_returns()
    volatile_in_scope = self.volatile_vars.pop()
    self.mark_safe_assignments(stmt.true, volatile_in_scope)
    self.mark_safe_assignments(stmt.false, volatile_in_scope)
  
  def is_mutable_alloc(self, expr):
    c = expr.__class__
    safe_types = (TupleT, ClosureT) 
    return c is syntax.Alloc or \
      c is syntax.Array or \
      c is syntax.ArrayView or \
      c is syntax.Slice or  \
      (c is syntax.Struct and not isinstance(expr.type, safe_types))
         
  def visit_Assign(self, stmt):
    lhs_names = collect_binding_names(stmt.lhs)
    rhs_names = collect_var_names(stmt.rhs)
    for x in lhs_names:
      dependencies = self.depends_on.get(x, set([]))
      dependencies.update(rhs_names)
      self.depends_on[x] = dependencies
      
    if any(x in self.volatile_vars for x in rhs_names) or \
        self.is_mutable_alloc(stmt.rhs):
      self.volatile_vars.update(lhs_names)
      
  def visit_fn(self, fn):

    self.volatile_vars.push(fn.arg_names)
    SyntaxVisitor.visit_fn(self, fn)
    return self.safe_to_move
   
from transform import Transform  

class LoopInvariantCodeMotion(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.analysis = Find_LICM_Candidates()
    self.safe_to_move = self.analysis.visit_fn(fn)
    self.binding_depth = {}
    # initially we have no blocks so need to offset by 1 
    # so that input names end up at depth 0 
    self.mark_binding_depths(fn.arg_names, 1)
  
  def mark_binding_depths(self, names, depth_offset = 0):
    curr_depth = len(self.blocks._blocks) - 1 + depth_offset
    for name in names:
      self.binding_depth[name] = curr_depth 
  

    
  def transform_While(self, stmt):
    self.mark_binding_depths(stmt.merge.iterkeys(), 1)
    return Transform.transform_While(self, stmt)
  
  def transform_If(self, stmt):
    self.mark_binding_depths(stmt.merge.iterkeys(), 1)
    return Transform.transform_If(self, stmt)
  
  def transform_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      name = stmt.lhs.name
      if name in self.safe_to_move:      
        deps = self.analysis.depends_on[name]
        if all(d in self.binding_depth for d in deps):
          if len(deps) > 0:
            target_level = max(self.binding_depth[d] for d in deps)
          else:
            target_level = 0  
          if target_level >= 0 and target_level < self.blocks.depth():
            self.blocks._blocks[target_level].append(stmt)
            self.binding_depth[name] = target_level 
            return None
    self.mark_binding_depths(collect_binding_names(stmt.lhs))
    return stmt 