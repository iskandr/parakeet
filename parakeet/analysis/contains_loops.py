from syntax_visitor import SyntaxVisitor

class Yes(Exception):
  pass 

class ContainsLoops(SyntaxVisitor):
  def visit_ForLoop(self, expr):
    raise Yes()
  
  def visit_While(self, expr):
    raise Yes()

def contains_loops(fn):
  try:
    ContainsLoops().visit_fn(fn)
    return False 
  except Yes:
    return True