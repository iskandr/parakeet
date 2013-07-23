from syntax_visitor import SyntaxVisitor

class FoundCall(Exception):
  pass

class ContainsCalls(SyntaxVisitor):
  def visit_Call(self, expr):
    raise FoundCall()

  def visit_fn(self, fn):
    try:
      self.visit_block(fn.body)
    except FoundCall:
      return True
    return False

def contains_calls(fn):
  return ContainsCalls().visit_fn(fn)