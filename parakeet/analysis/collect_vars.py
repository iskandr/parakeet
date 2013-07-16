from .. syntax import Var, Tuple
from syntax_visitor import SyntaxVisitor

class SetCollector(SyntaxVisitor):
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.var_names = set([])

  def visit_Var(self, expr):
    self.var_names.add(expr.name)

def collect_var_names(expr):
  collector = SetCollector()
  collector.visit_expr(expr)
  return collector.var_names

def collect_var_names_from_exprs(exprs):
  collector = SetCollector()
  collector.visit_expr_list(exprs)
  return collector.var_names

class ListCollector(SyntaxVisitor):
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.var_names = []

  def visit_Var(self, expr):
    self.var_names.append(expr.name)

def collect_var_names_list(expr):
  collector = ListCollector()
  collector.visit_expr(expr)
  return collector.var_names



def collect_binding_names(lhs):
  lhs_class = lhs.__class__
  if lhs_class is Var:
    return [lhs.name]
  elif lhs.__class__ is Tuple:
    combined = []
    for elt in lhs.elts:
      combined.extend(collect_binding_names(elt))
    return combined
  else:
    return []

class CollectBindings(SyntaxVisitor):
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.bindings = {}

  def bind(self, lhs, rhs):
    if lhs.__class__ is Var:
      self.bindings[lhs.name] = rhs
    elif lhs.__class__ is Tuple:
      for elt in lhs.elts:
        self.bind(elt, rhs)

  def visit_Assign(self, stmt):
    self.bind(stmt.lhs, stmt.rhs)

def collect_bindings(fn):
  return CollectBindings().visit_fn(fn)
