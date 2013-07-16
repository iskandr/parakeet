from syntax_visitor import SyntaxVisitor

from .. ndtypes  import TupleT, StructT, Type, PtrT, ClosureT, FnT

class TypeBasedMutabilityAnalysis(SyntaxVisitor):
  """
  The cheapest approximation to discovering which 
  fields are mutable is to do an insensitive analysis that
  marks all types which are on the RHS of a field assignment
  or are passed to a function (since other functions might 
  perform arbitrary mutable actions on fields). 
  
  Enter into the analysis via visit_fn, which returns 
  a set of struct types which may be mutable. 
  """
  
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.mutable_types = set([])
  
  def _mark_type(self, t):
    if self._has_mutable_fields(t):
      self.mutable_types.add(t)
      self._mark_children(t)
    elif isinstance(t, PtrT):
      self.mutable_types.add(t)
      

  def _has_mutable_fields(self, t):
    if isinstance(t, StructT):
      return not isinstance(t, (TupleT, ClosureT, FnT))
    
  def _mark_children(self, t):
    """
    For any type other than a Tuple or Closure, try 
    marking all its children 
    """
    for name in t.members():
      v = getattr(t, name)
      self._mark(v)
    if isinstance(t, StructT):
      for (_, field_type) in t._fields_:
        self._mark_type(field_type) 
  
      
  def _mark(self, obj):
    if isinstance(obj, Type):
      self._mark_type(obj)
    elif isinstance(obj, (tuple, list)):
      for child in obj:
        self._mark(child)
        
  def visit_merge(self, phi_nodes):
    pass 
    
  def visit_generic_expr(self, expr):
    pass 
  
  def visit_lhs_Tuple(self, expr):
    for e in expr.elts:
      self.visit_lhs(e)
  
  def visit_lhs_Attribute(self, expr):
    assert False, "Considering making attributes immutable"
    self._mark_type(expr.value.type)
  
  def visit_lhs_Index(self, expr):
 
    self._mark_type(expr.value.type)
  
  def visit_Call(self, expr):
    for arg in expr.args:
      """
      The fields of an argument type might change, 
      but since we pass by value the argument itself
      isn't made mutable
      """
      self._mark_type(arg.type)
    
  def visit_fn(self, fn):
    self.mutable_types.clear()
    self.visit_block(fn.body)
    return self.mutable_types

def find_mutable_types(fn):
  return TypeBasedMutabilityAnalysis().visit_fn(fn) 
