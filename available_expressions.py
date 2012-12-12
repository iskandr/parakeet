from syntax_visitor import SyntaxVisitor 
import core_types 
import tuple_type 
import closure_type 


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
  
  def __init__(self, fn):
    self.mutable_types = set([])
  
  def _mark_type(self, t):
    self.mutable_types.add(t)
    self._mark_children(t)

  def _has_mutable_fields(self, t):
    return isinstance(t, core_types.StructT) and \
        t.__class__ not in (tuple_type.TupleT, closure_type.ClosureT)
    
  def _mark_children(self, t):
    """
    For any type other than a Tuple or Closure, try 
    marking all its children 
    """
    if self._has_mutable_fields(t):
      for name in t.members():
        v = getattr(t, name)
        self._mark(v)
      if isinstance(t, core_types.StructT):
        for (_, field_type) in t._fields_:
          self._mark_type(field_type) 
  
      
  def _mark(self, obj):
    if isinstance(obj, core_types.Type):
      self._mark_type(obj)
    elif isinstance(obj, (tuple, list)):
      for child in obj:
        self._mark(child)
        
  def visit_generic_expr(self, expr):
    pass 
  
  def visit_lhs_Tuple(self, expr):
    for e in expr.elts:
      self.visit_lhs(e)
  
  def visit_lhs_Attribute(self, expr):
    assert expr.type 
    self._mark_type(expr.type)
  
  def visit_lhs_Index(self, expr):
    elt_type = expr.value.type.index_type(expr.index.type)
    self._mark_type(elt_type)
  
  def visit_lhs_TupleProj(self, expr):
    elt_type = expr.tuple.elt_types[expr.index]
    self._mark_type(elt_type)
    
  def visit_lhs_ClosureElt(self, expr):
    elt_type = expr.closure.arg_types[expr.index]
    self._mark_type(elt_type)
  
  def visit_Call(self, expr):
    for arg in expr.args:
      """
      The fields of an argument type might change, 
      but since we pass by value the argument itself
      isn't made mutable
      """
      self._mark_children(arg.type)
    
  def visit_fn(self, fn):
    self.mutable_types.clear()
    self.visit_block(fn.body)
    return self.mutable_types
    