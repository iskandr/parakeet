import syntax_visitor

class CollectVars(syntax_visitor.SyntaxVisitor):
  def __init__(self):
    self.var_names = set([])
    
  def visit_Var(self, expr):
    self.var_names.add(expr.name)
    
def collect_vars(expr):
  collector = CollectVars()
  collector.visit_expr(expr)
  return collector.var_names 