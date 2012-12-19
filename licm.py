import syntax 
from syntax_visitor import SyntaxVisitor 
from scoped_set import ScopedSet
from collect_vars import collect_vars

 
class Find_LICM_Candidates(SyntaxVisitor):
  def __init__(self):
    self.mutable_types = None
    self.volatile_vars = ScopedSet()
    self.depends_on = {}
    self.safe_to_move = set([])
  
  def mark_safe_assignments(self, block, volatile_set):
    for stmt in block:
      klass = stmt.__class__ 
      if klass is syntax.Assign and \
          stmt.lhs.__class__ is syntax.Var:
        name = stmt.lhs.name 
        dependencies = self.depends_on.get(name, set([]))
        if not any(d in volatile_set for d in dependencies):
          self.safe_to_move.add(name)
      else:
        # just in case there are Returns in nested control flow 
        # we should probably avoid changing the performance characteristics
        # by pulling out statements which will never run 
        break 
  
  def visit_Return(self, stmt):
    pass 
  
  def visit_merge(self, merge, both_branches = True):
    pass 
          
  def visit_While(self, stmt):
    self.volatile_vars.push(stmt.merge.keys())
    SyntaxVisitor.visit_While(self, stmt)
    volatile_in_scope = self.volatile_vars.pop()
    self.mark_safe_assignments(stmt.body, volatile_in_scope)
    
  def visit_If(self, stmt):
    self.volatile_vars.push(stmt.merge.keys())
    SyntaxVisitor.visit_If(self, stmt)
    volatile_in_scope = self.volatile_vars.pop()
    self.mark_safe_assignments(stmt.true, volatile_in_scope)
    self.mark_safe_assignments(stmt.false, volatile_in_scope)
    
  def visit_Assign(self, stmt):
    lhs_names = collect_vars(stmt.lhs)
    rhs_names = collect_vars(stmt.rhs)
    for x in lhs_names:
      dependencies = self.depends_on.get(x, set([]))
      dependencies.update(rhs_names)
      self.depends_on[x] = dependencies
    if any(x in self.volatile_vars for x in rhs_names):
      self.volatile_vars.update(lhs_names)
      
  def visit_fn(self, fn):

    self.volatile_vars.push(fn.arg_names)
    SyntaxVisitor.visit_fn(self, fn)
    return self.safe_to_move
   
from transform import Transform  

class LoopInvariantCodeMotion(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.safe_to_move = Find_LICM_Candidates().visit_fn(fn)
  
  def transform_Assign(self, stmt):
    if stmt.lhs.__class__ is syntax.Var and \
        stmt.lhs.name in self.safe_to_move:
      self.blocks._blocks[-2].append(stmt)
      return None 
    else:
      return stmt 