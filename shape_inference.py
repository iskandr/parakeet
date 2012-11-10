import syntax
from syntax_visitor import SyntaxVisitor 
from node import Node  

   
"""
class Max(ShapeElt):
  def __init__(self, d1, d2):
    self.d1 = d1
    self.d2 = d2
    
  def __eq__(self, other):
    return \
      (self.d1 == other.d1 and self.d2 == other.d2) or \
      (self.d2 == other.d1 and self.d1 == other.d1)  

def max_dim(d1, d2):
  if d1 == d2:
    return d1
  elif is_zero(d1):
    return d2 
  elif is_zero(d2):
    return d1 
  elif isinstance(d1, Const) and isinstance(d2, Const):
    if d1.value < d2.value:
      return d2 
    else:
      return d1 
  else:
    return Max(d1, d2)
"""



class ValueMismatch(Exception):
  """
  Raise this exception whenever two incompatible
  abstract values get combined
  """
  def __init__(self, v1, v2):
    self.v1 = v1
    self.v2 = v2 

  def __str__(self):
    return "ValueMismatch(%s, %s)" % (self.v1, self.v2)

  def __repr__(self):
    return str(self)
  
  
class AbstractValue(Node):
  pass 


class ConstValue(AbstractValue):
  def __init__(self, value):
    self.value = value 

def is_zero(d):
  return isinstance(d, ConstValue) and d.value == 0

def is_one(d):
  return isinstance(d, ConstValue) and d.value == 1

      
class ScalarVar(AbstractValue):
  def __init__(self, name):
    self.name = name 

class DimSize(AbstractValue):
  """
  The value producted by x.shape[d]
  """
  def __init__(self, var, dim):
    self.var = var
    self.dim = dim 
    
  def __eq__(self, other):
    return isinstance(other, DimSize) and \
      self.var == other.var and self.dim == other.dim
  
  def combine(self, other):
    if self == other:
      return self 
    else:
      raise ValueMismatch(self, other)

class ArrayValue(AbstractValue):
  def __init__(self, dims):
    self.dims = dims 
    self.rank = len(dims)
      
  def __eq__(self, other):
    return isinstance(other, ArrayValue) and \
      len(self.dims) == len(other.dims) and \
      all(d1 == d2 for (d1,d2) in zip(self.dims, other.dims) )
  
  def combine(self, other):
    if isinstance(other, ArrayValue):
      if is_scalar(other):
        return self
      elif is_scalar(self):
        return other
      elif other.rank == self.rank:
        
        assert False, "Shouldn't the logic of unifying dim variables be in the interpreter?"
    raise ValueMismatch(self, other)
  
# represent scalars with 0-rank array s
unknown_scalar = ArrayValue(())
    
def is_scalar(v):
  return isinstance(v, (ConstValue, ScalarVar, DimSize)) or \
    (isinstance(v, ArrayValue) and v.rank == 0)

class SliceValue(AbstractValue):
  def __init__(self, start, stop, step):
    self.start = start
    self.stop = stop 
    self.step = step 
    
  def __eq__(self, other):
    return isinstance(other, SliceValue) and \
      self.start == other.start and \
      self.stop == other.stop and \
      self.step == other.step 
  
  def combine(self, other):
    if isinstance(other, SliceValue):
      start = self.start.combine(other.start)
      stop = self.stop.combine(other.stop)
      step = self.step.combine(other.step)
      return SliceValue(start, stop, step)
    else:
      raise ValueMismatch(self, other)

def combine_value_lists(xs, ys):
  return [xi.combine(yi) for (xi, yi) in zip(xs, ys)]

def TupleValue(AbstractValue):
  def __init__(self, elts):
    self.elts = elts 
    
  def __eq__(self, other):
    return isinstance(other, TupleValue) and \
      len(self.elts) == len(other.elts) and \
      all(e1 == e2 for (e1, e2) in zip(self.elts, other.elts))
  
  def __str__(self):
    return "Tuple(%s)" % ", ".join(str(e) for e in self.elts)
  
  def combine(self, other):
    if isinstance(other, TupleValue):
      if len(self.elts) == len(other.elts):
        return TupleValue(combine_value_lists(self.elts, other.elts))
    raise ValueMismatch(self, other)
  
class ClosureValue(AbstractValue):
  def __init__(self, untyped_fn, args):
    self.untyped_fn = untyped_fn 
    self.args = args 
  
  def __str__(self):
    return "Closure(fn = %s, %s)" % \
      (self.untyped_fn, ", ".join(str(e) for e in self.elts))
  
  def __eq__(self, other):
    return isinstance(other, ClosureValue) and \
      self.untyped_fn == other.untyped_fn and \
      len(self.arg_shapes) == len(other.arg_shapes) and \
      all(v1 == v2 for (v1,v2) in zip(self.args, other.args))
  
  def combine(self, other):
    if isinstance(other, ClosureValue):
      # TODO: Implement sets of closures like we have in the type system 
      if self.untyped_fn == other.untyped_fn and \
         len(self.args) == len(other.args) :
        combined_args = combine_value_lists(self.args, other.args)
        return ClosureValue(self.untyped_fn, combined_args)
    raise ValueMismatch(self, other)  


class ShapeInference(SyntaxVisitor):
  
  def __init__(self):
    self.env = {}
    
  def visit_Const(self, expr):
    return ConstValue(expr.value)
  
  def visit_Var(self, expr):
    name = expr.name 
    assert name in self.env, "Unknown variable %s" % name
    return self.env[name]
  
  def visit_Tuple(self, expr):
    return TupleValue(self.visit_expr_list(expr.elts))

  def visit_Map(self, expr):
    arg_shapes = self.visit_expr_list(expr.args)
  
  def bind(self, lhs, rhs):
    if isinstance(lhs, syntax.Tuple)
  
  def visit_Assign(self, stmt):
    rhs = self.visit_expr(stmt.rhs)
    self.bind(self.lhs, rhs) 
      
  