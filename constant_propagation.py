class Const:
  
  def __init__(self, value):
    self.value = value 
    
  def merge(self, other):
    if self == other:
      return other
  
  def __equal__(self, other):
    return isinstance(other, Const) and self.value == other.value

class Top:
  pass
top_val = Top()    

import transform
import syntax

def match(lhs, rhs, env):
  if isinstance(lhs, syntax.Var) and isinstance(rhs, (syntax.Var, syntax.Const)):
    name = lhs.name
    if name in env:
      old_val = env[name]
      if old_val != rhs:
        env[name] = top_val 
    else:
      env[name] = rhs 
  elif isinstance(lhs, syntax.Tuple) and isinstance(rhs, syntax.Tuple):
    for (lhs_elt, rhs_elt) in zip(lhs.elts, rhs.elts):
      match(lhs_elt, rhs_elt, env)
        

class ConstantPropagation(transform.Transform):
  def __init__(self, fn):
    self.env = {}
    transform.Transform.__init__(self, fn)
    
  def transform_Var(self, expr):
    name = expr.name
    if name in self.env:
      val = self.env[name]
      if val != top_val:
        return val
    return expr
      
  def transform_Assign(self, stmt):
    match(stmt.lhs, stmt.rhs, self.env)

    return stmt     

def constant_propagation(fn):
  return ConstantPropagation(fn).apply()