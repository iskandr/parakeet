import escape_analysis

from array_type import ArrayT
from collect_vars import collect_var_names, collect_binding_names
from core_types import PtrT   
from syntax import Const, Var, Expr, Comment   
from syntax import Assign, Return, If, While, ForLoop 
 

node_counter = 0
def next_id():
  global node_counter
  node_counter += 1 
  return node_counter

class StmtNode:
  
  def __init__(self, stmt, consumes, produces, reads, writes, 
                stmt_id = None, scope = None):
    self.stmt = stmt 
    self.consumes = consumes 
    self.produces = produces 
    self.reads = reads 
    self.writes = writes 
    self.depends_on = set([])
    
    
    if stmt_id is None:
      self.id = next_id()
    else:
      self.id = stmt_id 
    self.scope = scope 
    
  def __hash__(self):
    return hash(id)
  
  def __eq__(self, other):
    return self.id == other.id
  
  def __str__(self):
    s = "%s(id = %d, scope = %s," % (self.stmt.__class__.__name__, self.id, self.scope)
    s += "\tdepends_on = %s,\n" % sorted(self.depends_on)
    s += "\tconsumes = %s,\n" % sorted(self.consumes)
    s += "\tproduces = %s,\n" % sorted(self.produces)
    s += "\treads = %s,\n" % sorted(self.reads.keys())
    s += "\twrites = %s)" % sorted(self.writes.keys())
    return s 
 
  def __repr__(self):
    return str(self)
    

class DependenceGraph(object):
  
  def __init__(self):
    self.produced_by = {}          
    self.returns = set([])
    self.nodes = set([])
    self.may_alias = None
  
  def visit_Var(self, expr, names, locations):
    names.add(expr.name)
  
  def visit_Call(self, expr, names, locations):
    arg_names = set([])
    self.visit_expr_list(expr.args, arg_names, locations)
    # have to assume that each function call might potentially
    # modify any mutable arguments 
    for name in arg_names:
      if isinstance(self.type_env[name], (ArrayT, PtrT)):
        locations.setdefault(name, set([])).add(expr)
    names.update(arg_names)
    
  def visit_Const(self, expr, names, locations):
    return                    
  
  def visit_Cast(self, expr, names, locations):
    self.visit_expr(expr.value, names, locations)
  
  def visit_Struct(self, expr, names, locations):
    self.visit_expr_list(expr.args, names, locations)
  
  def visit_Alloc(self, expr, names, locations):
    self.visit_expr(expr.count, names, locations)
  
  def visit_PrimCall(self, expr, names, locations):
    self.visit_expr_list(expr.args, names, locations)
  
  def visit_Tuple(self, expr, names, locations):
    self.visit_expr_list(expr.elts, names, locations)
  
  def visit_Attribute(self, expr, names, locations):
    assert expr.value.__class__ is Var, expr 
    names.add(expr.value.name)
    idx_set = locations.setdefault(expr.value.name, set([]))
    idx_set.add(expr)
  
  def visit_Index(self, expr, names, locations):
    assert expr.value.__class__ is Var, \
       "Invalid array in indexing expression %s" % expr 
    idx_names, idx_locations = self.visit_expr(expr.index)
    assert len(idx_locations) == 0, \
      "Shouldn't have recursive indexing in %s" % expr 
    names.update(idx_names)
    names.add(expr.value.name)
    idx_set = locations.setdefault(expr.value.name, set([]))
    idx_set.add(expr)
  
  def visit_expr_list(self, exprs, names, locations):
    for expr in exprs:
      self.visit_expr(expr, names, locations)
  
  def visit_expr(self, expr, names = None, locations = None):
    if names is None:
      names = set([])
    if locations is None:
      locations = {}
    c = expr.__class__ 
    if c is Var:
      names.add(expr.name)
      return names, locations 
    elif c is Const:
      return names, locations 

    method_name = 'visit_' + expr.node_type()
    if hasattr(self, method_name):
      method = getattr(self, method_name)
      method(expr, names, locations)
    else:
      for child in expr.children():                    
        if isinstance(child, Expr):
          self.visit_expr(child, names, locations)
    return names, locations 

  def visit_block(self, stmts):
    block_consumes = set([])
    block_reads = {}
    block_writes = {}
    last_wrote_to = {}
    produced_by = {}
    scope = id(stmts)
    
    for stmt in stmts:
      if stmt.__class__ is Comment:
        continue  
      node = self.visit_stmt(stmt, scope = scope)
      for consumed_name in node.consumes:
        block_consumes.add(consumed_name)
        for alias_name in self.may_alias.get(consumed_name, [consumed_name]):
          if alias_name in last_wrote_to: 
            pred_id = last_wrote_to[alias_name]
            node.depends_on.add(pred_id)
          elif alias_name in produced_by:
            pred_id = produced_by[alias_name]
            node.depends_on.add(pred_id)
          
      for (read_name, curr_read_set) in node.reads.iteritems():
        for alias_name in self.may_alias.get(read_name, [read_name]):
          if alias_name in last_wrote_to:
            pred_id = last_wrote_to[alias_name]
            node.depends_on.add(pred_id)
          elif alias_name in produced_by:
            pred_id = produced_by[alias_name]
            node.depends_on.add(pred_id)
        block_reads.setdefault(read_name, set([])).update(curr_read_set)
       
      for (write_name, curr_write_set) in node.writes.iteritems():
        for alias_name in self.may_alias.get(write_name, [write_name]):
          if alias_name in last_wrote_to:
            prev_writer_id = last_wrote_to[alias_name]
            node.depends_on.add(prev_writer_id)
        last_wrote_to[write_name] = node.id 
        block_writes.setdefault(write_name, set([])).update(curr_write_set)
        
      for produced_name in node.produces:
        produced_by[produced_name] = node.id

    return block_consumes, block_reads, block_writes
   
  def visit_merge(self, merge, consumes, produces):
    for (k,(l,r)) in merge.iteritems():                    
      produces.add(k)
      self.visit_expr(l, consumes, None)
      self.visit_expr(r, consumes, None) 
    
  def visit_Assign(self, stmt):
    consumes, reads = self.visit_expr(stmt.rhs)
    _, writes = self.visit_expr(stmt.lhs)
    for write_expr_set in writes.itervalues():
      for write_expr in write_expr_set:
        consumes.update(collect_var_names(write_expr))
    produces = collect_binding_names(stmt.lhs)

    return StmtNode(stmt,  consumes, produces, reads, writes)
  
  def visit_If(self, stmt):
    stmt_id = next_id()
    consumes_true, reads_true, writes_true = \
      self.visit_block(stmt.true)
    consumes_false, reads_false, writes_false = \
      self.visit_block(stmt.false)
    consumes = consumes_true.union(consumes_false)
    reads = reads_false 
    for (k,read_set) in reads_true.iteritems():
      reads_false.setdefault(k, set([])).update(read_set)
    writes = writes_false 
    for (k, write_set) in writes_true.iteritems():
      writes.setdefault(k, set([])).update(write_set)
    self.visit_expr(stmt.cond, consumes, reads)
    produces = set([])
    self.visit_merge(stmt.merge, consumes, produces)
    return StmtNode(stmt, consumes, produces, reads, writes,
                    stmt_id = stmt_id)
  
  def visit_While(self, stmt):
    stmt_id = next_id()
    consumes, reads, writes = \
      self.visit_block(stmt.body)
    self.visit_expr(stmt.cond, consumes, reads)
    produces = set([])
    self.visit_merge(stmt.merge, consumes, produces)
    return StmtNode(stmt, consumes, produces, reads, writes, 
                    stmt_id = stmt_id)
  
  def visit_ForLoop(self, stmt):
    stmt_id = next_id()
    produces = set([stmt.var.name])
    consumes, reads, writes = self.visit_block(stmt.body)
    self.visit_expr(stmt.start, consumes, reads)
    self.visit_expr(stmt.stop, consumes, reads)
    self.visit_expr(stmt.step, consumes, reads)
    self.visit_merge(stmt.merge, consumes, produces)
    return StmtNode(stmt, consumes, produces, reads, writes, 
                    stmt_id = stmt_id)
    
    
  
  def visit_Return(self, stmt):
    consumes, reads = self.visit_expr(stmt.value)
    self.returns.update(consumes)
    return StmtNode(stmt, consumes, set([]), reads, {})
  
  def visit_ExprStmt(self, stmt):
    assert False, \
       "Can't build dependence graphs with ExprStmt yet"
      
  
  def visit_stmt(self, stmt, scope = None):
    c = stmt.__class__ 
    if c is Assign:
      node = self.visit_Assign(stmt)
    elif c is If:
      node = self.visit_If(stmt)
    elif c is ForLoop:
      node = self.visit_ForLoop(stmt)
    elif c is While:
      node = self.visit_While(stmt)
    elif c is Return: 
      node = self.visit_Return(stmt)
    else:
      assert False, "Not implemented: %s" % stmt
    assert scope is not None
    node.scope = scope 
    self.nodes.add(node)
    return node 
  
  def visit_fn(self, fn):
    ee = escape_analysis.run(fn)
    self.may_alias = ee.may_alias
    
    self.may_escape = ee.may_escape 
    
    self.type_env = fn.type_env
    self.visit_block(fn.body)
