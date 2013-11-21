import numpy as np 

class AbstractValue(object):
  def __repr__(self):
    return str(self)
  
  def __hash__(self):
    assert False, "Hash function not implemented for %s : %s" % (self, self.__class__)
  
  def __eq__(self):
    return False 
  
  def __ne__(self, other):
    return not (self == other)

class Unknown(AbstractValue):
  def __str__(self):
    return "unknown"
  
  def __eq__(self, other):
    return other.__class__ is Unknown 
  
  def __hash__(self):
    return 107

unknown = Unknown()

class Tuple(AbstractValue):
  def __init__(self, elts):
    self.elts = tuple(elts)
    self._hash = hash(self.elts)
  
  def __str__(self):
    return "Tuple(%s)" % ", ".join(str(elt) for elt in self.elts)
  
  def __hash__(self):
    return self._hash 
  
  def __eq__(self, other):
    return other.__class__ is Tuple and self.elts == other.elts 
      
  
class Array(AbstractValue):
  # mark known strides with integer constants 
  # and all others as unknown
  def __init__(self, strides):
    self.strides = strides
    self._hash = hash(self.strides.elts) + 1
  
  def __str__(self):
    return "Array(strides = %s)" % self.strides
  
  def __hash__(self):
    return self._hash 
  
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


