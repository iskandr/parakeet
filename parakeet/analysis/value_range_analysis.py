from .. import prims  
from ..syntax import Var, Const, PrimCall
from syntax_visitor import SyntaxVisitor

import numpy as np 

class ValueRangeAnalyis(SyntaxVisitor):
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.ranges = {} 

  def get(self, expr):
    if expr.__class__ is Const:
      return (expr.value, expr.value)
    elif expr.__class__ is Var and expr.name in self.ranges:
      return self.ranges[expr.name]
    else:
      return None 
  
  def set(self, name, val):
    if val is not None:
      self.ranges[name] = val 

  def add(self, x, y):
    if x is None or y is None:
      return None 
    xl, xu = x
    yl, yu = y
    return (xl + yl, xu + yu)
  
  def sub(self, x, y):
    if x is None or y is None:
      return None 
    xl, xu = x
    yl, yu = y
    return (xl - yu, xu - yl)
  
  def mul(self, x, y):
    if x is None or y is None:
      return None 
    xl, xu = x
    yl, yu = y
    products = (xl * yl, xl * yu, xu * yl, xu * yu)
    lower = min(products)
    upper = max(products)
    return (lower, upper)
      
  def combine(self, x, y):
    if x is None or y is None:
      return None 
    xl, xu = x
    yl, yu = y 
    lower = min(xl, yl)
    upper = max(xu, yu)
    return (lower,upper)  
        
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      name = stmt.lhs.name
      if stmt.rhs.__class__ is PrimCall:
        p = stmt.rhs.prim 
        if p.nin == 2:
          x,y = stmt.rhs.args 
          if p == prims.add:
            self.set(name, self.add(self.get(x), self.get(y)))
          elif p == prims.subtract:
            self.set(name, self.sub(self.get(x), self.get(y)))
          elif p == prims.multiply:
            self.set(name, self.mul(self.get(x), self.get(y)))
      else:
        self.set(name, self.get(stmt.rhs))
  
  def visit_merge(self, phi_nodes):    
    for (k, (left,right)) in phi_nodes.iteritems():
      left_val = self.get(left)
      right_val = self.get(right)
      self.set(k, self.combine(left_val, right_val))

  def always_positive(self, x, inclusive = True):
    if x is None:
      return False 
    elif inclusive:
      return x[0] >= 0
    else:
      return x[0] > 0
  
  def always_negative(self, x, inclusive = True):
    if x is None:
      return False 
    elif inclusive:
      return x[1] <= 0
    else:
      return x[1] < 0
    
          
  def visit_ForLoop(self, stmt):
    start = self.get(stmt.start)
    stop = self.get(stmt.stop)
    step = self.get(stmt.step)
    name = stmt.var.name 
    if start is not None and stop is not None:
      start_val = stmt.start.value 
      stop_val = stmt.stop.value
      if start_val <= stop_val: 
        self.ranges[name] = (start_val, stop_val)
      else: 
        self.ranges[name] = (stop_val, start_val)
    
    elif start is not None and step is not None:
      if self.always_positive(step):
        self.ranges[name] = (start[0], np.inf)
      elif self.always_negative(step, inclusive = False):
        self.ranges[name] = (-np.inf, start[1])
    
    elif stop is not None and step is not None:
      if self.always_positive(step):
        self.ranges[name] = (-np.inf, stop[1])
      elif self.always_negative(step, inclusive = False):
        self.ranges[name] = (stop[0], np.inf)
    