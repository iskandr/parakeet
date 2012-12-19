import syntax_visitor

class Collector(syntax_visitor.SyntaxVisitor):
  def __init__(self):
    self.var_names = set([])
    
  def visit_Var(self, expr):
    self.var_names.add(expr.name)
    
def collect_var_names(expr):
  collector = Collector()
  collector.visit_expr(expr)
  return collector.var_names

from syntax import Var, Tuple 

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