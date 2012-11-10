
from args import match

class SyntaxVisitor(object):  
  def visit_generic_expr(self):
    pass
  
  def visit_expr(self, expr):
    method_name = 'vist_' + expr.node_type()
    if hasattr(self, method_name):
      method = getattr(self, method_name)
      return method(expr)
    else:
      return self.visit_generic_expr(expr)
  
  def visit_expr_list(self, exprs):
    return [self.visit_expr(e) for e in exprs]
  
  def visit_expr_tuple(self, exprs):
    return tuple(self.visit_expr_list(exprs))
  
  def visit_generic_stmt(self):
    pass
      
  def visit_stmt(self, stmt):
    method_name = 'vist_' + stmt.node_type()
    if hasattr(self, method_name):
      method = getattr(self, method_name)
      return method(stmt)
    else:
      return self.visit_generic_stmt(stmt) 

from node import Node  

class IncompatibleShapes(Exception):
  def __init__(self, s1, s2):
    self.s1 = s1
    self.s2 = s2 
    
  def __str__(self):
    return "IncompatibleShapes(%s, %s)" % (self.s1, self.s2)
  
  def __repr__(self):
    return str(self)


class ShapeElt(Node):
  pass 

class Const(ShapeElt):
  def __init__(self, value):
    self.value = value 
    
  def __eq__(self, other):
    return isinstance(other, Const) and \
      self.value == other.value 
      
def is_zero(d):
  return isinstance(d, Const) and d.value == 0

def is_one(d):
  return isinstance(d, Const) and d.value == 1

class DimSize(ShapeElt):
  def __init__(self, var, dim):
    self.var = var
    self.dim = dim 
    
  def __eq__(self, other):
    return self.var == other.var and self.dim == other.dim 

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
  
class SymbolicShape(Node):
  def __eq__(self, other):
    return False
  
  def combine_eq(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleShapes(self, other)
 
class UnknownShape(SymbolicShape):
  def combine_eq(self, other):
    return other
  
  def combine_max(self, other):
    return other 
    

class ArrayShape(SymbolicShape):
  def __init__(self, dims):
    self.dims = dims 
    
  def __eq__(self, other):
    return isinstance(other, ArrayShape) and \
      len(self.dims) == len(other.dims) and \
      all(d1 == d2 for (d1,d2) in zip(self.dims, other.dims) )
  
  def combine_max(self, other):
    if isinstance(other, ArrayShape):
      my_rank = len(self.dims)
      other_rank = len(other.dims)
      if my_rank == 0:
        return other
      elif other_rank == 0:
        return self
      elif my_rank == other_rank:
        return [max_dim(d1, d2) for (d1, d2) 
                in zip(self.arg_shapes, other.arg_shapes)]       
    raise IncompatibleShapes()

class ClosureShape(SymbolicShape):
  def __init__(self, untyped_fn, arg_shapes):
    self.untyped_fn = untyped_fn 
    self.arg_shapes = arg_shapes 
    
  def __eq__(self, other):
    return isinstance(other, ClosureShape) and \
      self.untyped_fn == other.untyped_fn and \
      len(self.arg_shapes) == len(other.arg_shapes) and \
      all(s1 == s2 for (s1,s2) in zip(self.arg_shapes, other.arg_shapes))

  def combine_max(self, other):
    if isinstance(other, ClosureShape) and \
      self.untyped_fn == other.untyped_fn and \
      len(self.arg_shapes) == len(other.arg_shapes):
        max_arg_shapes = \
          [s1.combine_max(s2) for (s1,s2) in
           zip(self.arg_shapes, other.arg_shapes)]
        return ClosureShape(self.untyped_fn, max_arg_shapes)
    raise IncompatibleShapes(self, other)


scalar_shape = ArrayShape(())
unknown_shape = UnknownShape()

def rank(self, s):
  assert isinstance(s, ArrayShape)
  return len(s.dims)
  
def is_scalar(self, s):
  return isinstance(s, ArrayShape) and rank(s) == 0 

def max_shape(shapes):
  s_max = unknown_shape
  for s in shapes:
    s_max = s_max.combine(s)
  return s_max 

def max_dims(shapes, axis):
  """
  TODO: skip scalars, slice into each array shape along the given axis
  """
  pass 

class ShapeInference(SyntaxVisitor):
  def visit_Const(self, _):
    return scalar_shape  
  
  def visit_Map(self, expr):
    arg_shapes = self.visit_expr_list(expr.args)
  
  def visit_Assign(self, stmt):
    shapes = self.visit_expr(stmt.rhs)
    match(self.lhs, shapes) 
      
  