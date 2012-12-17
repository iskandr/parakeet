from transform import Transform 
import syntax 
from mutability_analysis import TypeBasedMutabilityAnalysis
from scoped_env import ScopedEnv
class RedundancyElimination(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    ma = TypeBasedMutabilityAnalysis() 
    self.mutable_types = ma.visit_fn(fn)
    self.counter = 0 
    self.available_expressions = ScopedEnv() 
  
  def is_safe_tuple(self, e):
    return isinstance(e, syntax.Tuple) and \
        all(self.is_safe(elt) for elt in e.elts)
  
  def is_safe_closure(self, e):
    return isinstance(e, syntax.Closure) and \
        all(self.is_safe(arg) for arg in e.args)
  
  def is_safe_struct(self, e):
    return isinstance(e, syntax.Struct) and \
        e.type not in self.mutable_types and \
        all(self.is_safe(arg) for arg in e.args)
  
  def is_safe_create(self, e):
    return self.is_safe_tuple(e) or self.is_safe_closure(e) or \
        self.is_safe_struct(e)
         
  def is_safe_attr(self, e):
    return isinstance(e, (syntax.ClosureElt, syntax.TupleProj)) or \
           (isinstance(e, syntax.Attribute) and 
            e.value.type not in self.mutable_types)
           
  def is_safe_primcall(self, e):
    return isinstance(e, syntax.PrimCall) and \
        all(self.is_safe(a) for a in e.args)

  def is_safe(self, e):
    return isinstance(e, syntax.Const) or \
        (isinstance(e, syntax.Var) and 
         e.type not in self.mutable_types) or \
        self.is_safe_primcall(e) or \
        self.is_safe_create(e) or \
        self.is_safe_attr(e)
  
  def transform_expr(self, expr):
    if expr in self.available_expressions:
      self.counter += 1
      return self.available_expressions[expr]
    
    return Transform.transform_expr(self, expr)
  
  def transform_While(self, stmt):
    self.available_expressions.push()
    stmt = Transform.transform_While(self, stmt)
    self.available_expressions.pop()
    return stmt 
  
  def transform_If(self, stmt):
    self.available_expressions.push()
    stmt = Transform.transform_If(self, stmt)
    self.available_expressions.pop()
    return stmt 
  
  def transform_Assign(self, stmt):
    lhs = stmt.lhs 
    rhs = self.transform_expr(stmt.rhs)
    if isinstance(lhs, syntax.Var):
      print 
      print "LHS", lhs 
      print "RHS", rhs
      print "in avail?", rhs in self.available_expressions
      print "safe?", self.is_safe(rhs)
      for (k,v) in sorted(self.available_expressions.items()):
        print "    %s => %s" % (k,v)
        
      if rhs not in self.available_expressions \
          and (not isinstance(rhs, syntax.Var)) and self.is_safe(rhs):   
        self.available_expressions[rhs] = lhs
    return syntax.Assign(lhs, rhs)
  
  def pre_apply(self, fn):
    pass 
  
  def post_apply(self, fn):
    print "RedundancyElim removed %d expressions" % self.counter 
    # print repr(fn)