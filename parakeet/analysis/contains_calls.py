from syntax_visitor import SyntaxVisitor

class FoundCall(Exception):
  pass

class ContainsCalls(SyntaxVisitor):
  
  def visit_stmt(self, stmt):
    SyntaxVisitor.visit_stmt(self, stmt)
    
  def visit_Call(self, expr):
    raise FoundCall()

  def visit_fn(self, fn):
    try:
      self.visit_block(fn.body)
    except FoundCall:
      return True
    return False

def contains_calls(fn):
  result = ContainsCalls().visit_fn(fn)
  # print "Contains calls?", fn.name, result
  #if result: 
  #  print "==> ", fn  
  return result 