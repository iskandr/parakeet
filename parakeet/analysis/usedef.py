from .. syntax import Var, Tuple, Index, Attribute  
from syntax_visitor import SyntaxVisitor

class StmtPath(object):
  def __init__(self, elts = ()):
    self.elts = tuple(elts) 
  
  def __str__(self):
    return "{%s}" % ", ".join(str(e) for e in self.elts)
  
  def __len__(self):
    return len(self.elts)

  def __getitem__(self, i):
    return self.elts[i]
  
  def __iter__(self):
    return iter(self.elts)
  
  def __lt__(self, other):
    if other.__class__ is not StmtPath:
      other = StmtPath([other])
    n = len(other)
    for (i,c1) in enumerate(self.elts):
      if n <= i:
        return False 
      c2 = other[i]
      if type(c1) is int and type(c2) is int:
        if c1 < c2:
          return True 
        elif c1 > c2:
          return False 
      # if you're not on the same branch of control flow
      else:
        assert type(c1) is bool and type(c2) is bool 
        if c1 != c2:
          return False
    return False
  
  
  def __eq__(self, other):
    if other.__class__ is not StmtPath:
      other = StmtPath([other])
      
    if len(self) != len(other):
      return False
    else:
      for (i,c) in enumerate(self.elts):
        if c != other[i]:
          return False 
      return True 
  
  def __ne__(self, other):
    return not (self == other)
  
  def __le__(self, other):
    return self == other or self < other
  
  def __lte__(self, other):
    return (self < other) or (self == other)
  
  def __gt__(self, other):
    return not (self <= other) 
  
  def __gte__(self, other):
    return not (self < other)

class UseDefAnalysis(SyntaxVisitor):
  """
  Number all the statements and track the creation as well as first and last
  uses of variables
  """
  def __init__(self):
    # map from pointers of statement objects to
    # nested sequential numbering
    self.stmt_paths = {}

    # statement count per scope 
    self.stmt_counters = [0]
    
    # the scopes leading up to the current one 
    self.path_prefix = []
    
    
    # map from variable names to counter number of their
    # first/last usages
    self.first_use = {}
    self.last_use = {}
    self.first_read = {}
    self.last_read = {}
    self.first_write = {}
    self.last_write = {}
    self.created_on = {}
  
  def counter(self):
    return self.stmt_counters[-1]
  
  def inc_counter(self):
    self.stmt_counters[-1] += 1
  
  def push_counter(self):
    self.stmt_counters.append(0)
    
  def pop_counter(self):
    return self.stmt_counters.pop()
  
  def push_scope(self, c = None):
    if c is None:
      c = self.counter()
    self.path_prefix.append(c) 
  
  def pop_scope(self):
    self.path_prefix.pop()
  
  def curr_path(self):
    return StmtPath(self.path_prefix + [self.counter()])

  def precedes_stmt_id(self, stmt_id1, stmt_id2):
    return self.stmt_paths[stmt_id1] < self.stmt_paths[stmt_id2] 
    
  def precedes_stmt(self, stmt1, stmt2):
    return self.stmt_paths[id(stmt1)] < self.stmt_paths[id(stmt2)] 
    
  def visit_fn(self, fn):
    for name in fn.arg_names:
      self.created_on[name] = 0
    SyntaxVisitor.visit_fn(self, fn)

  def visit_lhs(self, expr):
    if expr.__class__ is Var:
      self.created_on[expr.name] = self.curr_path()
    elif expr.__class__ is Tuple:
      for elt in expr.elts:
        self.visit_lhs(elt)
    else:
      assert expr.__class__ in (Attribute, Index)
      assert expr.value.__class__ is Var
      value_name = expr.value.name 
      p = self.curr_path()
      if value_name not in self.first_write:
        self.first_write[value_name] = p 
      self.last_write[value_name] = p

  def vist_merge(self, merge):
    p = self.curr_path()
    for (name, (l,r)) in merge.iteritems():
      self.visit_expr(l)
      self.visit_expr(r)
      self.created_on[name] = p
  
  def visit_Var(self, expr):
    name = expr.name
    p = self.curr_path()
    if name not in self.first_use:
      self.first_use[name] = p
    self.last_use[name]= p

  def visit_block(self, stmts, branch_label = None):
    self.push_scope()
    if branch_label is not None:
      self.push_scope(branch_label)
    SyntaxVisitor.visit_block(self, stmts)
    if branch_label is not None:
      self.pop_scope()
    self.pop_scope()
    
  def visit_If(self, stmt):
    self.visit_expr(stmt.cond)
    self.visit_block(stmt.true, True)
    self.visit_block(stmt.false, False)
    self.visit_merge(stmt.merge)
    
  def visit_stmt(self, stmt):
    self.inc_counter()
    self.stmt_paths[id(stmt)] = self.curr_path()
    SyntaxVisitor.visit_stmt(self, stmt)
