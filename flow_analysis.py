from syntax import Var 
from syntax_visitor import SyntaxVisitor 

empty = set([])
class FlowAnalysis(SyntaxVisitor):
  def __init__(self):
    """
    Identify values/objects with the id of syntax nodes. 
    Analyze function to determine:
      1) which values may/must flow into which variables
      2) which variables may/must be pointing to a particular value
      3) which variables may/must return from the function 
      4) which values may/must return from the function 
      5)   
    """
    self.var_may_point_to_values = {}
    self.var_must_point_to_values = {}
    
    # need to track nested references along fields
    self.value_may_point_to_values = {}
    
    self.value_may_be_aliased_by_vars = {}
    self.value_must_be_aliased_by_vars = {}
    
    self.vars_must_return = set([])
    self.vars_may_return = set([])
    self.values_may_return = set([])
    self.values_must_return = set([])
    
    self.var_may_escape = set([])
    self.values_may_escape = set([])
  
  def visit_Return(self, stmt):
    if stmt.rhs.__class__ is Var:
      name = stmt.rhs.name 
      self.vars_may_return.add(name)
      self.vars_must_return.add(name)
    else:
      rhs_id = id(stmt.rhs)
      self.values_must_return.add(rhs_id)
      self.values_may_return.add(rhs_id)
      self.values_may_return.extend(self.visit_expr(stmt.rhs))
  
  def visit_Var(self, expr):
    return self.var_may_point_to_values.get(expr.name, empty)
  
  def visit_merge(self, merge):
    for (name, (l,r)) in merge.iteritems():
      left_set = self.visit_expr(l)
      right_set = self.visit_expr(r)
      combined_set = left_set.union(right_set)
      for value_id in combined_set:
        if value_id in self.values_may_be_aliased_by_vars:
          self.values_may_be_aliased_by_vars[value_id].add(name)
        else:
          self.values_may_be_aliased_by_vars[value_id] = set([name])
      self.var_may_point_to_values[name] = combined_set 
      
        
  