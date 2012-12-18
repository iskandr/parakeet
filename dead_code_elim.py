import syntax
from common import dispatch
from transform import Transform
import syntax_helpers 
from syntax_visitor import SyntaxVisitor 

class FindLiveVars(SyntaxVisitor):
  def __init__(self):
    self.live_vars = set([])
    
  def visit_Var(self, expr):
    self.live_vars.add(expr.name)
    
  def visit_merge(self, merge, both_branches=True):
    for (_, (l,r)) in merge.iteritems():
      self.visit_expr(l)
      self.visit_expr(r) 
      
    

  def visit_lhs(self, expr):
    if isinstance(expr, syntax.Var):
      pass 
    elif isinstance(expr, syntax.Tuple):
      for elt in expr.elts:
        self.visit_lhs(elt)
    else:
      self.visit_expr(expr)
      
  def visit_fn(self, fn):
    self.live_vars.clear()
    for name in fn.arg_names:
      self.live_vars.add(name)
    self.visit_block(fn.body)
    return self.live_vars
    

class DCE(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.live_vars = FindLiveVars().visit_fn(fn)
  
    
  def is_live_lhs(self, lhs):
    if isinstance(lhs, syntax.Var):
      return lhs.name in self.live_vars
    elif isinstance(lhs, syntax.Tuple):
      return any(self.is_live_lhs(elt) for elt in lhs.elts)
    elif isinstance(lhs, str):
      return lhs in self.live_vars
    elif isinstance(lhs, tuple):
      return any(self.is_live_lhs(elt) for elt in lhs)
    else:
      return True

  def transform_phi_nodes(self, phi_nodes):
    new_merge = {}
    for var in phi_nodes:
      if var in self.live_vars:
        new_merge[var] = phi_nodes[var]
    return new_merge
  
  def transform_Assign(self, stmt):
    if self.is_live_lhs(stmt.lhs):
      return stmt
    else:
      return None

  def transform_While(self, stmt):
    # expressions don't get changed by this transform
    cond = stmt.cond
    new_body = self.transform_block(stmt.body) 
    new_merge = self.transform_phi_nodes(stmt.merge)
    if len(new_merge) == 0 and len(new_body) == 0:
      return None
    else:
      return syntax.While(cond, new_body, new_merge)

  def transform_If(self, stmt):
    cond = stmt.cond 
    new_true = self.transform_block(stmt.true) 
    new_false = self.transform_block(stmt.false)
    new_merge = self.transform_phi_nodes(stmt.merge)
    if len(new_merge) == 0 and len(new_true) == 0 and len(new_false) == 0:
      return None  
    elif syntax_helpers.is_true(cond):
      self.blocks.extend_current(stmt.true)
      for name, (_, v) in new_merge.items():
        self.assign(syntax.Var(name, type = v.type), v)
      return None 
    elif syntax_helpers.is_false(cond):
      self.blocks.extend_current(stmt.false)
      for name, (v, _) in new_merge.items():
        self.assign(syntax.Var(name, type = v.type), v)
      return None 
    else:
      return syntax.If(cond, new_true, new_false, new_merge)

  def transform_Return(self, stmt):
    return stmt
  
  def post_apply(self, fn):
    fn.type_env = \
      dict([(name, fn.type_env[name]) for name in self.live_vars])
    Transform.post_apply(self, fn)
    return fn 
  
def dead_code_elim(fn):
  dce = DCE(fn)
  return dce.apply(copy = False)
  