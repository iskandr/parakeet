import numpy as np

import type_conv

from array_type import ArrayT
from core_types import StructT
from dead_code_elim import DCE
from node import Node
from pipeline_phase import Phase
from simplify import Simplify
from syntax import Var
from syntax_helpers import const_int
from syntax_visitor import SyntaxVisitor
from transform import Transform
from tuple_type import TupleT

class AbstractValue(object):
  def __repr__(self):
    return str(self)
  
  def __hash__(self):
    return hash(str(self))

class Unknown(AbstractValue):
  def __str__(self):
    return "unknown"

unknown = Unknown()

class Tuple(AbstractValue):
  def __init__(self, elts):
    self.elts = tuple(elts)
  
  def __str__(self):
    return "Tuple(%s)" % ", ".join(str(elt) for elt in self.elts)
  
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
    assert isinstance(value, int)
    self.value = value
  
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

def from_internal_repr(parakeet_type, v):
  if v is None:
    return unknown
  elif hasattr(v, 'contents'):
    v = v.contents
    
  if parakeet_type.__class__ is TupleT:
    elts = []
    for (i,elt_t) in enumerate(parakeet_type.elt_types):
      elt_value = hasattr(v, "elt%d" % i)
      elts.append(from_internal_repr(elt_t, elt_value))
    return abstract_tuple(elts)
  elif parakeet_type.__class__ is ArrayT:
    strides_field = getattr(v, 'strides').contents
    strides = []
    for i in xrange(parakeet_type.rank):
      s = int(getattr(strides_field, 'elt%d'%i))
      strides.append(specialization_const(s))
    return abstract_array(strides)  
  elif isinstance(parakeet_type, StructT):
    fields = {}
    for (field_name, field_type) in parakeet_type._fields_:
      field_value = getattr(v, field_name)
      abstract_field = from_internal_repr(field_type, field_value)
      fields[field_name] = abstract_field
    return Struct(fields)        
  return unknown

def from_python(python_value):
  if isinstance(python_value, np.ndarray):
    elt_size = python_value.dtype.itemsize 
    strides = []
    for s in python_value.strides:
      strides.append(specialization_const(s/elt_size)) 
    return abstract_array(strides)
  elif isinstance(python_value, tuple):
    return abstract_tuple(from_python_list(python_value))
  else:
    parakeet_type = type_conv.typeof(python_value)
    parakeet_value = type_conv.from_python(python_value)
    return from_internal_repr(parakeet_type, parakeet_value)
  
def from_python_list(python_values):
  return [from_python(v) for v in python_values] 

def bind_list(arg_names, abstract_values):
  assert len(arg_names) == len(abstract_values)
  env = {}
  for name, abstract_value in zip(arg_names, abstract_values):
    env[name] = abstract_value
  return env 

class FindConstantStrides(SyntaxVisitor):
  def __init__(self, fn, abstract_values):
    self.env = bind_list(fn.arg_names, abstract_values)
  
  def visit_Var(self, expr):
    return self.env.get(expr.name, unknown)
  
  def visit_Const(self, expr):
    if expr.value == 0:
      return zero
    elif expr.value == 1:
      return one 
    elif isinstance(expr.value, int):
      return Const(expr.value)
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
      if value is not None:
        self.env[stmt.lhs.name] = value
      else:
        print "%s returned None in ConstantStride analysis" % stmt.rhs
  
  
class StrideSpecializer(Transform):
  def __init__(self, abstract_inputs):
    Transform.__init__(self)
    self.abstract_inputs = abstract_inputs 
  
  def pre_apply(self, fn):
    analysis = FindConstantStrides(fn, self.abstract_inputs)
    analysis.visit_fn(fn)
    self.env = analysis.env
  
  def transform_Var(self, expr):
    if expr.name in self.env:
      value = self.env[expr.name]
      if value.__class__ is Const:
        return const_int(value.value)
    return expr
  
  def transform_lhs(self, lhs):
    return lhs
  
def has_unit_stride(abstract_value):
  c = abstract_value.__class__
  if c is Array:
    return has_unit_stride(abstract_value.strides)
  elif c is Struct:
    return any(has_unit_stride(field_val) 
               for field_val 
               in abstract_value.fields.itervalues())
  elif c is Tuple:
    return any(has_unit_stride(elt) 
               for elt in abstract_value.elts)
  elif c is Const:
    return abstract_value.value == 1
  else:
    return False
  
_cache = {}
def specialize(fn, python_values, types = None):
  if types is None:
    abstract_values = from_python_list(python_values)
  else:
    # if types are given, assume that the values 
    # are already converted to Parakeet's internal runtime 
    # representation 
    abstract_values = []
    for (t, internal_value) in zip(types, python_values):
      abstract_values.append(from_internal_repr(t, internal_value))
  
  key = (fn.name, tuple(abstract_values))
  if key in _cache:
    return _cache[key]
  elif any(has_unit_stride(v) for v in abstract_values):
    specializer = StrideSpecializer(abstract_values)
    
    transforms = Phase([specializer, Simplify, DCE],
                        memoize = False, copy = True)
    new_fn = transforms.apply(fn)
  else:
    new_fn = fn
  _cache[key] = new_fn
  return new_fn