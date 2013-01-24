"""
Generic traversal of a tree-structured type
"""

class Traversal(object):
  def visit_list(self, xs):
    return [self.visit(x) for x in xs]
  
  def visit_tuple(self, xs):
    return tuple(self.visit_list(xs))
  
  def visit_generic(self, x):
    assert False, \
      "Unsupported %s : %s" % (x, x.__class__.__name__)
  
  def visit(self, x):
    method_name = 'visit_' + x.__class__.__name__
    
    if hasattr(self, method_name):
      method = getattr(self, method_name)
      return method(x)
    else:
      return self.visit_generic(x)
      