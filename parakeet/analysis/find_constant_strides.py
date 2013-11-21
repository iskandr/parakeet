
import numpy as np 

from .. ndtypes import ArrayT, TupleT   
from .. syntax import TypedFn, Var
from syntax_visitor import SyntaxVisitor

class AbstractValue(object):
  def __repr__(self):
    return str(self)
  
  def __hash__(self):
    return hash(str(self))
  
  def __eq__(self):
    return False 
  
  def __ne__(self, other):
    return not (self == other)

class Unknown(AbstractValue):
  def __str__(self):
    return "unknown"
  
  def __eq__(self, other):
    return other.__class__ is Unknown 

unknown = Unknown()

class Tuple(AbstractValue):
  def __init__(self, elts):
    self.elts = tuple(elts)
  
  def __str__(self):
    return "Tuple(%s)" % ", ".join(str(elt) for elt in self.elts)
  
  def __hash__(self):
    return hash(self.elts)
  
  def __eq__(self, other):
    return other.__class__ is Tuple and \
      len(self.elts) == len(other.elts) and \
      all(e1 == e2 for (e1,e2) in zip(self.elts, other.elts))
  
class Array(AbstractValue):
  # mark known strides with integer constants 
  # and all others as unknown
  def __init__(self, strides):
    self.strides = strides
  
  def __str__(self):
    return "Array(strides = %s)" % self.strides
  
  def __hash__(self):
    return hash(self.strides.elts)
  
  def __eq__(self, other):
    return other.__class__ is Array and \
      self.strides == other.strides

class Struct(AbstractValue):
  def __init__(self, fields):
    self.fields = fields 
    
  def __str__(self):
    return "Struct(%s)" % self.fields
  
  def __eq__(self, other):
    if other.__class__ != Struct:
      return False
    my_fields = set(self.fields.keys())
    other_fields = set(other.fields.keys())
    if my_fields != other_fields:
      return False 
    for f in my_fields:
      if self.fields[f] != other.fields[f]:
        return False
    return True
  
  def __hash__(self):
    return hash(tuple(self.fields.items()))
      
class Const(AbstractValue):
  def __init__(self, value):
    assert isinstance(value, (int, long, bool, np.bool_, np.ScalarType)), \
                      "Expected scalar but got %s : %s" % (value, type(value)) 
    self.value = value
  
  def __hash__(self):
    return int(self.value) 
  
  def __str__(self):
    return str(self.value)
  
  def __eq__(self, other):
    return other.__class__ is Const and self.value == other.value

zero = Const(0)
one = Const(1)

def specialization_const(x, specialize_all = False):
  if x == 0:
    return zero 
  elif x == 1:
    return one
  elif specialize_all:
    return Const(x)
  else:
    return unknown

def abstract_tuple(elts):
  return Tuple(tuple(elts))

def abstract_array(strides):
  return Array(abstract_tuple(strides))



def bind_list(arg_names, abstract_values):
  assert len(arg_names) == len(abstract_values)
  env = {}
  for name, abstract_value in zip(arg_names, abstract_values):
    env[name] = abstract_value
  return env 

class FindConstantStrides(SyntaxVisitor):
  def __init__(self, fn, abstract_values):
    self.env = bind_list(fn.arg_names, abstract_values)
    self.return_value = None 
    
  def visit_expr(self, expr):
    result = SyntaxVisitor.visit_expr(self, expr)
    if result is None: 
      return unknown
    else: 
      return result 
  
  def visit_fn(self, fn):
    SyntaxVisitor.visit_fn(self, fn)
    if self.return_value is None:
      return unknown 
    else:
      return self.return_value
  
  def visit_Call(self, expr):
    if expr.fn.__class__ is TypedFn: 
      args = self.visit_expr_list(expr.args)
      return  FindConstantStrides(expr.fn, args).visit_fn(expr.fn)
    else:
      return unknown 

  def visit_Var(self, expr):
    return self.env.get(expr.name, unknown)
  
  def visit_Const(self, expr):
    if expr.value == 0:
      return zero
    elif expr.value == 1:
      return one 
    elif isinstance(expr.value, int):
      return Const(int(expr.value))
    else:
      return unknown
  
  def visit_Attribute(self, expr):
    value = self.visit_expr(expr.value)
    if value.__class__ is Tuple:
      # assume tuples lowered into structs at this point
      pos = expr.value.type.field_pos(expr.name)
      return value.elts[pos]
    elif value.__class__ is Array and expr.name == 'strides':
      return value.strides
    elif value.__class__ is Struct and expr.name in value.fields:
      return value.fields[expr.name]
    else:
      return unknown

  def visit_TupleProj(self, expr):
    value = self.visit_expr(expr.tuple)
    if isinstance(value, Tuple):
      return value.elts[expr.index]
    else:
      return unknown 
    
  def visit_PrimCall(self, expr):
    abstract_values = self.visit_expr_list(expr.args)
    if all(v.__class__ is Const for v in abstract_values):
      ints = [v.value for v in abstract_values]
      return Const(expr.prim.fn(*ints))
    else:
      return unknown

  def visit_Struct(self, expr):
    if expr.type.__class__ is TupleT:
      return abstract_tuple(self.visit_expr_list(expr.args))
    elif expr.type.__class__ is ArrayT:
      stride_pos = expr.type.field_pos("strides")
      stride_arg = expr.args[stride_pos]
      stride_val = self.visit_expr(stride_arg)
      return Array(stride_val)
    else:
      return unknown   
    
  def visit_Alloc(self, expr):
    return unknown 
  
  def visit_Index(self, expr):
    return unknown 
  
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      value = self.visit_expr(stmt.rhs)
      assert value is not None, \
        "%s : %s returned None in ConstantStride analysis" % (stmt.rhs, stmt.rhs.node_type()) 
      self.env[stmt.lhs.name] = value 
      
  def visit_Return(self, stmt):
    value = self.visit_expr(stmt.value)
    if self.return_value is None:
      self.return_value = value 
    elif self.return_value != value:
      self.return_value = unknown 