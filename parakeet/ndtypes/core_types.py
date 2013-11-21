import ctypes

#from dsltools import Node

class TypeFailure(Exception):
  def __init__(self, msg):
    self.msg = msg

class IncompatibleTypes(Exception):
  def __init__(self, t1, t2):
    self.t1 = t1
    self.t2 = t2

  def __repr__(self):
    return "IncompatibleTypes(%s, %s)" % (self.t1, self.t2)

  def __str__(self):
    return repr(self)

class Type(object):
  def combine(self, other):
    raise IncompatibleTypes(self, other)

  def children(self):
    return ()
  
  def node_type(self):
    return self.__class__.__name__
  
  def __hash__(self):
    assert False, "Hash function not implemented for type %s" % (self,)

  def __eq__(self, _):
    assert False, "Equality not implemented for type %s" % (self,)
  
  def __ne__(self, other):
    return not (self == other)
  
  def __str__(self):
    return self.__class__.__name__

class AnyT(Type):
  """top of the type lattice, absorbs all types"""

  def __hash__(self):
    return 101
  
  def combine(self, other):
    return self

  def __eq__(self, other):
    return other.__class__ is AnyT

# since there's only one Any type, just create an instance of the same name
Any = AnyT()

class UnknownT(Type):
  """Bottom of the type lattice, absorbed by all other types"""

  _hash = hash("unknown")
  
  def __hash__(self):
    return self._hash 
  
  
  _members = []
  def  combine(self, other):
    return other

  def __eq__(self, other):
    return other.__class__ is UnknownT

#single instance of the Unknown type with same name
Unknown = UnknownT()

def combine_type_list(types):
  common_type = Unknown

  for t in types:
    common_type = common_type.combine(t)
  return common_type



class ConcreteT(Type):
  """
  Type which actually have some corresponding runtime values, as opposed to
  "Any" and "Unknown"
  """

  def from_python(self, py_val):
    return self.ctypes_repr(py_val)

  def to_python(self, internal):
    return internal
  
class ImmutableT(ConcreteT):
  """
  Tuples, closures, None, and scalars have no mutable fields
  """
  pass 
# None is fully described by its type, so the
# runtime representation can just be the number zero

class NoneT(ImmutableT):
  _members = []
  rank = 0
  ctypes_repr = ctypes.c_int64


  def combine(self, other):
    if isinstance(other, NoneT):
      return self
    else:
      raise IncompatibleTypes(self, other)

  def __str__(self):
    return "NoneT"

  def __hash__(self):
    return 0

  def __eq__(self, other):
    return other.__class__ is NoneT
  
  def __repr__(self):
    return str(self)

NoneType = NoneT()



def is_struct(c_repr):
  return type(c_repr) == type(ctypes.Structure)

class FieldNotFound(Exception):
  def __init__(self, struct_t, field_name):
    self.struct_t = struct_t
    self.field_name = field_name
    
  def __str__(self):
    return "FieldNotFound(%s, %s)" % (self.struct_t, self.field_name)

class StructT(Type):
  """All concrete types excluding scalars and pointers"""

  # expect each child class to fill this list
  _fields_ = []

  _repr_cache = {}

  def field_type(self, name):
    for (field_name, field_type) in self._fields_:
      if field_name == name:
        return field_type
    raise FieldNotFound(self, name)

  def field_pos(self, name):
    for (i, (field_name, _)) in enumerate(self._fields_):
      if field_name == name:
        return i
    raise FieldNotFound(self, name)

###################################################
#                                                 #
#             SCALAR NUMERIC TYPES                #
#                                                 #
###################################################

###################################################
# helper functions to implement properties of Python scalar objects
###################################################
def always_zero(x):
  return type(x)(0)

def identity(x):
  return x

class TypeValueT(ImmutableT):
  """
  Materialization of a type into a value 
  """
  rank = 0
  
  def __init__(self, type):
    self.type = type 
    self._hash = hash(self.type)

  
  def __str__(self):
    return "TypeValue(%s)" % self.type 
  
  def __hash__(self):
    return self._hash 
  
  def __eq__(self, other):
    return other.__class__ is TypeValueT and self.type == other.type
  
