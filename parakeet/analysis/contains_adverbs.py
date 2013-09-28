
from syntax_visitor import SyntaxVisitor
from .. syntax import Adverb 

class ContainsAdverbs(SyntaxVisitor):
  class Yes(Exception):
    pass
  
  def visit_expr(self, expr):
    if isinstance(expr, Adverb):
      raise self.Yes()
    SyntaxVisitor.visit_expr(self, expr)
  

def contains_adverbs(fn):
  try:
    ContainsAdverbs().visit_fn(fn)
  except ContainsAdverbs.Yes:
    return True
  return False