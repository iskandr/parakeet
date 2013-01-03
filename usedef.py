from syntax import Var, Tuple 
from syntax_visitor import SyntaxVisitor

class UseDefAnalysis(SyntaxVisitor):
  """
  Number all the statements and track the creation as well as first and last
  uses of variables
  """
  def __init__(self):
    # map from pointers of statement objects to
    # sequential numbering
    # where start is the current statement
    self.stmt_number = {}

    # ..and also track the range of nested statement numbers
    self.stmt_number_end = {}

    self.stmt_counter = 0
    # map from variable names to counter number of their
    # first/last usages
    self.first_use = {}
    self.last_use = {}
    self.created_on = {}

  def visit_fn(self, fn):
    for name in fn.arg_names:
      self.created_on[name] = 0
    SyntaxVisitor.visit_fn(self, fn)

  def visit_lhs(self, expr):
    if expr.__class__ is Var:
      self.created_on[expr.name] = self.stmt_counter
    elif expr.__class__ is Tuple:
      for elt in expr.elts:
        self.visit_lhs(elt)

  def visit_If(self, stmt):
    for name in stmt.merge.iterkeys():
      self.created_on[name] = self.stmt_counter
    SyntaxVisitor.visit_If(self, stmt)

  def visit_While(self, stmt):
    for name in stmt.merge.iterkeys():
      self.created_on[name] = self.stmt_counter
    SyntaxVisitor.visit_While(self, stmt)

  def visit_Var(self, expr):
    name = expr.name
    if name not in self.first_use:
      self.first_use[name] = self.stmt_counter
    self.last_use[name]= self.stmt_counter

  def visit_stmt(self, stmt):
    stmt_id = id(stmt)
    self.stmt_counter += 1
    count = self.stmt_counter
    self.stmt_number[stmt_id] = count
    SyntaxVisitor.visit_stmt(self, stmt)
    self.stmt_number_end[stmt_id] = self.stmt_counter
