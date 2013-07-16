from .. import  prims 
from .. syntax import Var, PrimCall, Const
from syntax_visitor import SyntaxVisitor

class OffsetAnalysis(SyntaxVisitor):
  """Determine static offset relationships between variables"""
  
  def __init__(self):
    # map from variable 'x' to list of variable offset pairs
    # [ ('y', 4), ('z', -1), ...] 
    self.known_offsets = {}
    
  def update(self, x, y, k):
    if x == y:
      assert k == 0, \
         "Impossible %s = %s + %d" % (x,y,k)
    
    if x in self.known_offsets:
      x_offsets = self.known_offsets[x]
    else:
      x_offsets = set([])
      
    x_offsets.add( (y,k) )

    if y in self.known_offsets:
      y_offsets = self.known_offsets[y]
    else:
      y_offsets = set([])
      
    for (z, k2) in y_offsets:
      x_offsets.add( (z, k2 + k) )
      
    self.known_offsets[x] = x_offsets
    self.known_offsets[y] = y_offsets
    
  def visit_merge(self, merge):
    for (k, (l,r)) in merge.iteritems():
      if l.__class__ is Var and \
         r.__class__ is Var and \
         l.name in self.known_offsets and \
         r.name in self.known_offsets:
        left = self.known_offsets[l.name]
        right = self.known_offsets[r.name]
        self.known_offsets[k] = left.intersection(right)  
  
  def visit_PrimCall(self, expr):
    if expr.prim is prims.add:
      x, y = expr.args
      if x.__class__ is Var and y.__class__ is Const:
        return (x.name, y.value)
      elif y.__class__ is Var and x.__class__ is Const:
        return (y.name, x.value)
    elif expr.prim is prims.subtract:
      x, y = expr.args
      if x.__class__ is Var and y.__class__ is Const:
        return (x.name, -y.value)
    return None
  
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      if stmt.rhs.__class__ is PrimCall:
        rhs = self.visit_PrimCall(stmt.rhs)
        if rhs is not None:
          x = stmt.lhs.name
          (y, offset) = rhs
          self.update(x, y, offset)
          self.update(x, y, offset)
          self.update(y, x, -offset)
      elif stmt.rhs.__class__ is Var:
        x = stmt.lhs.name 
        y = stmt.rhs.name
        self.update(x, y, 0)
        self.update(y, x, 0)
  
  def visit_fn(self, fn):
    SyntaxVisitor.visit_fn(self, fn)
    return self.known_offsets

