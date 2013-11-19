from dsltools import  ScopedDict
 

from .. import prims  
from ..ndtypes import NoneT, NoneType, get_rank 
from ..syntax import Var, Const, PrimCall, Tuple, Slice, TupleProj, Attribute, Shape 
 
import numpy as np 
from ..syntax.helpers import unwrap_constant
from syntax_visitor import SyntaxVisitor



class AbstractValue(object):
  def __eq__(self, other):
    return False    

  def __ne__(self, other):
    return not self == other
  
  def __repr__(self):
    return str(self)
  
  def combine(self, other):
    if self == other:
      return self 
    else:
      return any_value 
  
  def widen(self, other):
    if self == other:
      return self 
    else:
      assert False, "widening not implemented for (%s, %)" % (self, other) 
    
  
class Unknown(AbstractValue):
  """
  Top of the lattice 
  """
  
  def __str__(self):
    return "<unknown>"
  
  def combine(self, other):
    return other
  
  def widen(self, other):
    return other  
  
  
unknown_value = Unknown()

class AnyValue(AbstractValue):
  """
  Bottom of the lattice 
  """
  
  def __str__(self):
    return "<any>"
  
  def combine(self, other):
    return any_value 
  
  def widen(self, other):
    return any_value 
  
  def __eq__(self, other):
    return other.__class__ is AnyValue 
  
any_value = AnyValue()

class NoneValue(AbstractValue):
  def __str__(self):
    return "None"
  
  
  def __eq__(self, other):
    return other.__class__ is NoneValue
  
  def combine(self, other):
    if self == other:
      return self 
    else:
      return any_value 
    
  def widen(self, other):
    return self.combine(other)

class Interval(AbstractValue):
  def __init__(self, lower, upper):
    self.lower = lower 
    self.upper = upper 
  
  def combine(self, other):
    if other.__class__ is not Interval:
      return any_value 
    
    lower = min(self.lower, other.lower)
    upper = max(self.upper, other.upper)
    return Interval(lower,upper)
    
  def __eq__(self, other):
    return other.__class__ is Interval and self.lower == other.lower and self.upper == other.upper
  
  def widen(self, other):
    
    """
    Like 'combine' but not symmetric-- the other value
    is a more recent version from within a loop, 
    so try to follow the 'trend' of its changes. 
    """
    
    if other.__class__ is not Interval:
      return any_value 
    lower_diff = other.lower - self.lower
    upper_diff = other.upper - self.upper
    

    if lower_diff < 0:
      lower = -np.inf
    else:
      lower = self.lower 
    
    
    
    if upper_diff > 0:
      upper = np.inf 
    else:
      upper = self.upper 
    
    if lower == self.lower and upper == self.upper:
      return self 
    else:
      return Interval(lower, upper)     
    
  def __str__(self):
    return "[%s,%s]" % (self.lower, self.upper)
    
const_zero = Interval(0,0)
const_one = Interval(1,1)
const_none = NoneValue()
positive_interval = Interval(0, np.inf)
negative_interval = Interval(-np.inf, 0)
  
class TupleOfIntervals(AbstractValue):
  def __init__(self, elts):
    assert all(isinstance(elt, AbstractValue) for elt in elts)
    self.elts = elts 
    
  def combine(self, other):
    if other.__class__ is not TupleOfIntervals:
      return any_value 
    elif len(other.elts) != len(self.elts):
      return any_value  
    combined_elts = [e1.combine(e2) for e1,e2 in zip(self.elts, other.elts)]
    return TupleOfIntervals(combined_elts)

  def widen(self, other):
    if other.__class__ is not TupleOfIntervals or len(other.elts) != len(self.elts):
      return any_value 
    return mk_tuple([e1.widen(e2) for e1,e2 in zip(self.elts, other.elts)])
  
  def __str__(self):
    return "tuple(%s)" % ", ".join(str(elt) for elt in self.elts)

  def __eq__(self, other):
    return other.__class__ is TupleOfIntervals and len(other.elts) == len(self.elts) and \
      all(e1 == e2 for e1,e2 in zip(self.elts,other.elts))

def mk_tuple(elts):
  if all(elt is None or elt is any_value for elt in elts):
    return any_value 
  else:
    return TupleOfIntervals(elts)  

class SliceOfIntervals(AbstractValue):
  def __init__(self, start, stop, step):
    self.start = start 
    self.stop = stop 
    self.step = step 
    
  def combine(self, other):
    if other.__class__ is not SliceOfIntervals:
      return any_value
    start = self.start.combine(other.start)
    stop = self.stop.combine(other.stop)
    step = self.step.combine(other.step)
    if start == self.start and stop == self.stop and step == self.stop:
      return self 
    return mk_slice(start,stop,step)

  def __eq__(self, other):
    return other.__class__ is SliceOfIntervals and \
      self.start == other.start and \
      self.stop == other.stop and \
      self.step == other.step

  def widen(self, other):
    if other.__class__ is not SliceOfIntervals:
      return any_value 
    start = self.start.widen(other.start)
    stop = self.stop.widen(other.stop)
    step = self.step.widen(other.step)
    if start == self.start and stop == self.stop and step == self.step:
      return self 
    return mk_slice(start,stop,step)
        
  def __str__(self):
    return "slice(start=%s, stop=%s, step=%s)" % (self.start, self.stop, self.step)
    
def mk_slice(start, stop, step):
  if (start is None or start is any_value) and \
     (stop is None or stop is any_value)  and \
     (step is None or step is any_value):
    return any_value 
  return SliceOfIntervals(start, stop, step) 



class ValueRangeAnalyis(SyntaxVisitor):
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.ranges = {} 
    self.old_values = ScopedDict()
    self.old_values.push()

  def get(self, expr):
    c = expr.__class__ 
    
    if expr.type.__class__ is NoneT:
      return const_none 
    elif c is Const:
      return Interval(expr.value, expr.value)
    
    elif c is Var and expr.name in self.ranges:
      return self.ranges[expr.name]
    
    elif c is Tuple:
      elt_values = [self.get(elt) for elt in expr.elts]
      return mk_tuple(elt_values)
    
    elif c is TupleProj:
      tup = self.get(expr.tuple)
      idx = unwrap_constant(expr.index) 
      if tup.__class__ is TupleOfIntervals:
        return tup.elts[idx]
      
    elif c is Slice:
      start = self.get(expr.start)
      stop = self.get(expr.stop)
      if expr.step.type ==  NoneType:
        step = const_one   
      else:
        step = self.get(expr.step)
      return mk_slice(start, stop, step)
    
    elif c is Shape:
      ndims = get_rank(expr.array.type)   
      return mk_tuple([positive_interval] * ndims)
    
    elif c is Attribute:
      if expr.name == 'shape':
        ndims = get_rank(expr.value.type)   
        return mk_tuple([positive_interval] * ndims)
      elif expr.name == 'start':
        sliceval = self.get(expr.value)
        if isinstance(sliceval, SliceOfIntervals):
          return sliceval.start
      elif expr.name == 'stop':
        sliceval = self.get(expr.value)
        if isinstance(sliceval, SliceOfIntervals):
          return sliceval.stop
      elif expr.name == 'step':
        sliceval = self.get(expr.value)
        if isinstance(sliceval, SliceOfIntervals):
          return sliceval.step
    elif c is PrimCall:
      p = expr.prim 
      if p.nin == 2:
        x = self.get(expr.args[0])
        y = self.get(expr.args[1])
        if p == prims.add:

          return self.add_range(x,y)
        elif p == prims.subtract:
          return self.sub_range(x,y)
        elif p == prims.multiply:
          return self.mul_range(x,y)
      elif p.nin == 1:
        x = self.get(expr.args[0])
        if p == prims.negative:
          return self.neg_range(x)
    return any_value 
  
  def set(self, name, val):
    if val is not None and val is not unknown_value:
      old_value = self.ranges.get(name, unknown_value)
      self.ranges[name] = val
      if old_value != val and old_value is not unknown_value:
        self.old_values[name] = old_value 

      
  def add_range(self, x, y):
    if not isinstance(x, Interval) or not isinstance(y, Interval):
      return any_value 
    return Interval (x.lower + y.lower, x.upper + y.upper)
  
  def sub_range(self, x, y):
    if not isinstance(x, Interval) or not isinstance(y, Interval):
      return any_value
    return Interval(x.lower - y.upper, x.upper - y.lower)
  
  def mul_range(self, x, y):
    if not isinstance(x, Interval) or not isinstance(y, Interval):
      return any_value
    xl, xu = x.lower, x.upper 
    yl, yu = y.lower, y.upper 
    products = (xl * yl, xl * yu, xu * yl, xu * yu)
    lower = min(products)
    upper = max(products)
    return Interval(lower, upper)
  
  def neg_range(self, x):
    if not isinstance(x, Interval):
      return any_value 
    return Interval(-x.upper, -x.lower) 
  
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      name = stmt.lhs.name 
      v = self.get(stmt.rhs)
      self.set(name, v)
  
  def visit_merge_left(self, phi_nodes):
    for (k, (left, _)) in phi_nodes.iteritems():
      left_val = self.get(left)
      self.set(k, left_val)
      
      
  def visit_merge(self, phi_nodes):    
    for (k, (left,right)) in phi_nodes.iteritems():
      left_val = self.get(left)
      right_val = self.get(right)
      self.set(k, left_val.combine(right_val))

  def visit_Select(self, expr):
    return self.get(expr.true_value).combine(self.get(expr.false_value))
  
  def always_positive(self, x, inclusive = True):
    if not isinstance(x, Interval):
      return False 
    elif inclusive:
      return x.lower >= 0
    else:
      return x.lower > 0
  
  def always_negative(self, x, inclusive = True):
    if not isinstance(x, Interval):
      return False 
    elif inclusive:
      return x.upper <= 0
    else:
      return x.upper < 0
    
  
  def widen(self, old_values):
    for (k, oldv) in old_values.iteritems():
      newv = self.ranges[k]
      if oldv != newv:
        self.ranges[k] = oldv.widen(newv)
          
  def run_loop(self, body, merge):
    
    
    # run loop for the first time 
    self.old_values.push()
    self.visit_merge_left(merge)
    self.visit_block(body)
    self.visit_merge(merge)
    
    #run loop for the second time 
    self.visit_block(body)
    self.visit_merge(merge)
    
    old_values = self.old_values.pop()
    self.widen(old_values)
    
    # TODO: verify that it's safe not to run loop with widened values
    #self.visit_block(body)
    #self.visit_merge(merge)
    
    
  def visit_While(self, stmt):
    self.run_loop(stmt.body, stmt.merge)
  
  
  def visit_ForLoop(self, stmt):    

    start = self.get(stmt.start)
    stop = self.get(stmt.stop)
    step = self.get(stmt.step)
    name = stmt.var.name
    iterator_range = any_value  
    if isinstance(start, Interval) and isinstance(stop, Interval):
      lower = min (start.lower, start.upper, stop.lower, stop.upper)
      upper = max (start.lower, start.upper, stop.lower, stop.upper)
      iterator_range = Interval(lower,upper)
      
    elif isinstance(start, Interval) and isinstance(step, Interval):
      if self.always_positive(step):
        iterator_range = Interval(start.lower, np.inf)
      elif self.always_negative(step, inclusive = False):
        iterator_range = Interval(-np.inf, start.upper)
    
    elif isinstance(stop, Interval) and isinstance(step, Interval):
      if self.always_positive(step):
        iterator_range = Interval(-np.inf, stop.upper)
      elif self.always_negative(step, inclusive = False):
        iterator_range = Interval(stop.lower, np.inf)  
    self.set(name, iterator_range)
    self.run_loop(stmt.body, stmt.merge)

      
    