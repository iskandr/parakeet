
from syntax_visitor import SyntaxVisitor
from .. syntax import Adverb, ParFor 

class ContainsAdverbs(SyntaxVisitor):
  class Yes(Exception):
    pass
  
  def visit_expr(self, expr):
    if isinstance(expr, Adverb):
      raise self.Yes()
    SyntaxVisitor.visit_expr(self, expr)
  
  def visit_stmt(self, stmt):
    if isinstance(stmt, ParFor):
      raise self.Yes()
    else:
      SyntaxVisitor.visit_stmt(self, stmt)
    
def contains_adverbs(fn):
  try:
    ContainsAdverbs().visit_fn(fn)
  except ContainsAdverbs.Yes:
    return True
  return False