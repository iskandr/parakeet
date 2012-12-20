import syntax 

class SyntaxVisitor(object):  
  """
  Traverse the statement structure of a syntax block,
  optionally collecting values 
  """
  def visit_generic_expr(self, expr):
    
    for v in expr.itervalues():
      if v and isinstance(v, syntax.Expr):
        self.visit_expr(v)
      elif isinstance(v, (list,tuple)):
        for child in v:
          if isinstance(child, syntax.Expr):
            self.visit_expr(child)
            
  def visit_expr(self, expr):
    method_name = 'visit_' + expr.node_type()

    if hasattr(self, method_name):
      method = getattr(self, method_name)
      return method(expr)
    else:
      return self.visit_generic_expr(expr)
  
  def visit_expr_list(self, exprs):
    return [self.visit_expr(e) for e in exprs]
  
  def visit_expr_tuple(self, exprs):
    return tuple(self.visit_expr_list(exprs))
  
  def visit_lhs(self, lhs):
    method_name = 'visit_lhs_' + lhs.node_type()
    if hasattr(self, method_name):
      method = getattr(self, method_name)
      return method(lhs)
    return self.visit_expr(lhs)
  
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
  
  def visit_merge_loop_start(self, phi_nodes):
    pass 
  
  def visit_merge_loop_repeat(self, phi_nodes):
    return self.visit_merge(phi_nodes)
  
  def visit_merge_if(self, phi_nodes):
    return self.visit_merge(phi_nodes)
    
    
  def visit_If(self, stmt):
    self.visit_block(stmt.true)
    self.visit_block(stmt.false)
    self.visit_merge_if(stmt.merge)
    self.visit_expr(stmt.cond)
    
  def visit_Return(self, stmt):
    self.visit_expr(stmt.value)
    
  def visit_While(self, stmt):
    self.visit_merge_loop_start(stmt.merge)
    self.visit_expr(stmt.cond)
    self.visit_block(stmt.body)
    self.visit_merge_loop_repeat(stmt.merge)
    
    
  def visit_stmt(self, stmt):
    method_name = 'visit_' + stmt.node_type()
    if hasattr(self, method_name):
      method = getattr(self, method_name)
      return method(stmt)
    else:
      return self.visit_generic_stmt(stmt) 

  def visit_fn(self, fn):
    return self.visit_block(fn.body)
