from syntax_visitor import SyntaxVisitor
from ..ndtypes import ScalarT, TupleT

class Yes(Exception):
  pass 

class ContainsSlices(SyntaxVisitor):
  def visit_Index(self, expr):
    if isinstance(expr.index.type, ScalarT):
      idx_types = [expr.index.type]
    elif isinstance(expr.index.type, TupleT):
      idx_types = expr.index.type.elt_types
    else: 
      raise Yes()
    if not all(isinstance(t, ScalarT) for t in idx_types):
      raise Yes()
  
def contains_slices(fn):
  try:
    ContainsSlices().visit_fn(fn)
    return False 
  except Yes:
    return True