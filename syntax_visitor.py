import syntax 

class SyntaxVisitor(object):  
  """
  Traverse the statement structure of a syntax block,
  optionally collecting values 
  """
  def visit_generic_expr(self, expr):
    pass 

  def visit_expr(self, expr):
    method_name = 'visit_' + expr.node_type()
    print method_name 
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
    return self.visit_expr(lhs)
  
  def after_Assign(self, new_lhs, new_rhs):
    """
    Gets called after children are visited
    """
    pass
  
  def after_While(self, new_cond, new_body, new_merge):
    pass
  
  def after_Return(self, new_value):
    pass
  
  def after_If(self, new_cond, new_true, new_false, new_merge):
    pass
   
  def after_block(self, stmts):
    pass
  
  def visit_block(self, stmts):
    # only try to collect the returned result 
    # for each statement if the descendant class
    # has define some meaningful operation that 
    # receives all these statements 
    if self.after_block.im_class != SyntaxVisitor: 
      new_stmts = [self.visit_stmt(s) for s in stmts]
      return self.after_block(new_stmts)
    else:
      for s in stmts:
        self.visit_stmt(s)
  
  def visit_generic_stmt(self, stmt):
    if isinstance(stmt, syntax.Assign):
      lhs = self.visit_lhs(stmt.lhs)
      rhs = self.visit_expr(stmt.rhs)
      return self.after_Assign(lhs, rhs)
    elif isinstance(stmt, syntax.If):
      cond = self.visit_expr(stmt.cond)
      true = self.visit_block(stmt.true)
      false = self.visit_block(stmt.false)
      merge = self.visit_merge(stmt.merge)
      return self.after_If(cond, true, false, merge)
    elif isinstance(stmt, syntax.Return):
      value = self.visit_expr(stmt.value)
      return self.after_Return(value)
    else:
      assert isinstance(stmt, syntax.While), \
        "Unexpected statement: " + str(stmt)
      cond = self.visit_expr(stmt.cond)
      body = self.visit_block(stmt.body)
      merge = self.visit_merge(stmt.merge)
      return self.after_While(cond, body, merge)
    
    
  def visit_stmt(self, stmt):
    method_name = 'visit_' + stmt.node_type()
    if hasattr(self, method_name):
      method = getattr(self, method_name)
      return method(stmt)
    else:
      return self.visit_generic_stmt(stmt) 

  def visit_fn(self, fn):
    return self.visit_block(fn.body)
