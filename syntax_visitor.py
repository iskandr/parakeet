import syntax 
from syntax import Assign, Return, If, While 
from syntax import Var, Const, Tuple, Index, Attribute, PrimCall

class SyntaxVisitor(object):  
  """
  Traverse the statement structure of a syntax block,
  optionally collecting values 
  """
  
  def visit_Var(self, expr):
    pass
   
  def visit_Const(self, expr):
    pass 
  
  def visit_Tuple(self, expr):
    self.visit_expr_list(expr.elts)
  
  def visit_PrimCall(self, expr):
    self.visit_expr_list(expr.args)
  
  def visit_Attribute(self, expr):
    self.visit_expr(expr.value)
    
  def visit_Index(self, expr):
    self.visit_expr(expr.value)
    self.visit_expr(expr.index)
    
  def visit_expr(self, expr):
    c = expr.__class__ 
    if c is Var:
      return self.visit_Var(expr)
    elif c is Const:
      return self.visit_Const(expr)
    elif c is Tuple:
      return self.visit_Tuple(expr)
    elif c is Index:
      return self.visit_Index(expr)
    elif c is PrimCall:
      return self.visit_PrimCall(expr)
    elif c is Attribute:
      return self.visit_Attribute(expr)
    else:
      
      method_name = 'visit_' + expr.node_type()
      method = getattr(self, method_name, None)
      if method:
        return method(expr)
      else:
        for v in expr.itervalues():
          if v and isinstance(v, syntax.Expr):
            self.visit_expr(v)
          elif isinstance(v, (list,tuple)):
            for child in v:
              if isinstance(child, syntax.Expr):
                self.visit_expr(child)
            
  def visit_expr_list(self, exprs):
    return [self.visit_expr(expr) for expr in exprs]
  
  def visit_lhs_Var(self, lhs):
    self.visit_Var(lhs)

  def visit_lhs_Tuple(self, lhs):
    self.visit_Tuple(lhs)
  
  def visit_lhs_Index(self, lhs):
    self.visit_Index(lhs)
  
  def visit_lhs_Attribute(self, lhs):
    self.visit_Attribute(lhs)
  
  def visit_lhs(self, lhs):
    c = lhs.__class__ 
    if c is Var:
      return self.visit_lhs_Var(lhs)
    elif c is Tuple:
      return self.visit_lhs_Tuple(lhs)
    elif c is Index:
      return self.visit_lhs_Index(lhs)
    elif c is Attribute:
      return self.visit_lhs_Attribute(lhs)
    else:
      assert False, "LHS not implemented: %s" % (lhs,)
      
  def visit_block(self, stmts):
    for s in stmts:
      self.visit_stmt(s)
  
  def visit_Assign(self, stmt):
    self.visit_lhs(stmt.lhs)
    self.visit_expr(stmt.rhs)
  
  def visit_merge(self, phi_nodes):
    for (_, (l,r)) in phi_nodes.iteritems():
      self.visit_expr(l)
      self.visit_expr(r)
        
  def visit_merge_if(self, phi_nodes):
    self.visit_merge(phi_nodes)
    
  def visit_If(self, stmt):
    self.visit_expr(stmt.cond)
    self.visit_block(stmt.true)
    self.visit_block(stmt.false)
    self.visit_merge_if(stmt.merge)
    
  def visit_Return(self, stmt):
    self.visit_expr(stmt.value)
  
  def visit_merge_loop_start(self, phi_nodes):
    pass 
  
  def visit_merge_loop_repeat(self, phi_nodes):
    self.visit_merge(phi_nodes)
    
  def visit_While(self, stmt):
    self.visit_merge_loop_start(stmt.merge)
    self.visit_expr(stmt.cond)
    self.visit_block(stmt.body)
    self.visit_merge_loop_repeat(stmt.merge)
    
    
  def visit_stmt(self, stmt):
    c = stmt.__class__ 
    if c is Assign: 
      self.visit_Assign(stmt)
    elif c is Return:
      self.visit_Return(stmt)
    elif c is While:
      self.visit_While(stmt)
    elif c is If:
      self.visit_If(stmt)
    else:
      assert False, "Statement not implemented: %s" % stmt      
    
  def visit_fn(self, fn):
    self.visit_block(fn.body)
