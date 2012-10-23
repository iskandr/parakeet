import syntax 
import names 



class NestedBlocks:
  def __init__(self):
    self.blocks = []
     
  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    return self.blocks.pop()
  
  def current(self):
    return self.blocks[-1]
  
  def append_to_current(self, stmt):
    self.current().append(stmt)
  
  def extend_current(self, stmts):
    self.current().extend(stmts)


class Transform:
  def __init__(self, fn):
    self.blocks = NestedBlocks()
    self.fn = fn
    self.type_env = None 
  
  
  def lookup_type(self, name):
    assert self.type_env is not None
  
  def fresh_var(self, t, prefix = "temp"):
    ssa_id = names.fresh(prefix)
    self.type_env[ssa_id] = t
    return syntax.Var(ssa_id, t)
  
  def insert_stmt(self, stmt):
    self.blocks.append_to_current(stmt)
  
  def transform_generic_expr(self, expr):
    args = {}
    for member_name in expr._members:
      member_value = getattr(expr, member_name)
      if isinstance(member_value, syntax.Expr):
        member_value = self.transform_expr(member_value)
      args[member_name] = member_value
    return expr.__class__(**args)
      
  
  def transform_expr(self, expr):
    """
    Dispatch on the node type and call the appropriate transform method
    """
    method_name = "transform_" + expr.node_type()
    if hasattr(self, method_name):
      return getattr(self, method_name)(expr)
    else:
      return self.transform_generic_expr(expr)
  
  def transform_expr_list(self, exprs):
    return [self.transform_expr(e) for e in exprs]
  
  def transform_phi_nodes(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      result[k] = new_left, new_right 
    return result 
  
  def transform_Assign(self, stmt):
    # TODO: flatten tuple assignment ptype
    assert isinstance(stmt.lhs, (str, syntax.Var)), \
      "Pattern-matching assignment not implemented" 
    rhs = self.transform_expr(stmt.rhs)
    return syntax.Assign(stmt.lhs, rhs)
    
  def transform_Return(self, stmt):
    return syntax.Return(self.transform_expr(stmt.value))
    
  def transform_If(self, stmt):
    cond = self.transform_expr(stmt.cond)
    true = self.transform_block(stmt.true)
    false = self.transform_block(stmt.false)
    merge = self.transform_phi_nodes(stmt.merge)      
    return syntax.If(cond, true, false, merge) 
    
  def transform_While(self, stmt):
    cond = self.transform_expr(stmt.cond)
    body = self.transform_block(stmt.body)
    merge_before = self.transform_phi_nodes(stmt.merge_before)
    merge_after = self.transform_phi_nodes(stmt.merge_after)
    return syntax.While(cond, body, merge_before, merge_after)
  

  
  def transform_stmt(self, stmt):
    method_name = "transform_" + stmt.node_type()
    if hasattr(self, method_name):
      getattr(self, method_name)(stmt)
    
  def transform_block(self, stmts):
    self.blocks.push()
    for old_stmt in stmts:
      new_stmt = self.transform_stmt(old_stmt)
      self.blocks.append_to_current(new_stmt)
    return self.blocks.pop() 
  
  def apply(self):
    self.type_env = self.fn.type_env.copy()
    body = self.transform_block(self.fn.body)
    new_fundef_args = dict([ (m, getattr(self.fn, m)) for m in self.fn._members])
    new_fundef_args['body'] = body
    new_fundef_args['type_env'] = self.type_env 
    return syntax.TypedFn(**new_fundef_args)
