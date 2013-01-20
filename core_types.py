import ctypes
import numpy as np

import dtypes
import type_conv

from node import Node

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

class Type(Node):
  def combine(self, other):
    raise IncompatibleTypes(self, other)

  def __hash__(self):
    assert False, "Hash function not implemented for type %s" % (self,)

  def __eq__(self, _):
    assert False, "Equality not implemented for type %s" % (self,)

class AnyT(Type):
  """top of the type lattice, absorbs all types"""

  def combine(self, other):
    return self

  def __eq__(self, other):
    return isinstance(other, AnyT)

# since there's only one Any type, just create an instance of the same name
Any = AnyT()

class UnknownT(Type):
  """Bottom of the type lattice, absorbed by all other types"""

  _members = []
  def  combine(self, other):
    return other

  def __eq__(self, other):
    return isinstance(other, UnknownT)

#single instance of the Unknown type with same name
Unknown = UnknownT()

class FnT(Type):
  """Type of a typed function"""

  def __init__(self, input_types, return_type):
    self.input_types = tuple(input_types )
    self.return_type = return_type

  def __str__(self):
    input_str = ", ".join(str(t) for t in self.input_types)
    return "(%s)->%s" % (input_str, self.return_type)

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return isinstance(other, FnT) and \
           self.return_type == other.return_type and \
           len(self.input_types) == len(other.input_types) and \
           all(t1 == t2 for (t1, t2) in
               zip(self.input_types, other.input_types))

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)

  def __hash__(self):
    return hash(self.input_types + (self.return_type,))

_fn_type_cache = {}
def make_fn_type(input_types, return_type):
  input_types = tuple(input_types)
  key = input_types, return_type
  if key in _fn_type_cache:
    return _fn_type_cache[key]
  else:
    t = FnT(input_types, return_type)
    _fn_type_cache[key] = t
    return t

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

  def ctypes_repr(self):
    pass

  def from_python(self, py_val):
    return self.ctypes_repr(py_val)

  def to_python(self, internal):
    return internal

# None is fully described by its type, so the
# runtime representation can just be the number zero

class NoneT(ConcreteT):
  _members = []
  ctypes_repr = ctypes.c_int64

  def from_python(self, val):
    assert val is None
    return ctypes.c_int64(0)

  def to_python(self, obj):
    assert obj == ctypes.c_int64(0), \
        "Expected runtime representation of None to be 0, but got: %s" % obj
    return None

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
    return isinstance(other, NoneT)
  def __repr__(self):
    return str(self)

NoneType = NoneT()
def typeof_none(_):
  return NoneType
type_conv.register(type(None), NoneT, typeof_none)

def is_struct(c_repr):
  return type(c_repr) == type(ctypes.Structure)

class FieldNotFound(Exception):
  def __init__(self, struct_t, field_name):
    self.struct_t = struct_t
    self.field_name = field_name

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

  @property
  def ctypes_repr(self):
    if self in self._repr_cache:
      return self._repr_cache[self]

    ctypes_fields = []

    for (field_name, parakeet_field_type) in self._fields_:
      field_repr = parakeet_field_type.ctypes_repr
      # nested structures will be heap allocated
      if isinstance(parakeet_field_type,  StructT):
        ptr_t = ctypes.POINTER(field_repr)
        ctypes_fields.append( (field_name, ptr_t) )
      else:
        ctypes_fields.append( (field_name, field_repr) )

    class Repr(ctypes.Structure):
      _fields_ = ctypes_fields
    Repr.__name__ = self.node_type() +"_Repr"
    self._repr_cache[self] = Repr
    return Repr

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

# base class for all concrete scalar types
# don't actually tag any values with this
class ScalarT(ConcreteT):
  rank = 0
  _members = ['dtype']

  # no need for _fields_ since scalars aren't represented as a struct
  _properties_ = [
    ('real', identity),
    ('imag', always_zero),
    ('conjugate', identity)
  ]

  def node_init(self):
    assert isinstance(self.dtype, np.dtype), \
        "Expected dtype, got %s" % self.dtype
    self.name = self.dtype.name
    self.nbytes = self.dtype.itemsize

  @property
  def ctypes_repr(self):
    return dtypes.to_ctypes(self.dtype)

  def __eq__(self, other):
    return other.__class__ is self.__class__ and self.nbytes == other.nbytes
    
  def __hash__(self):
    return hash(self.dtype)

  def __repr__(self):
    return self.name

  def __str__(self):
    return str(self.name)

  def index_type(self, _):
    return self

  def combine(self, other):
    if isinstance(other, ScalarT):
      combined_dtype = np.promote_types(self.dtype, other.dtype)
      if combined_dtype == self.dtype:
        return self
      elif combined_dtype == other.dtype:
        return other
      else:
        return from_dtype(combined_dtype)
    else:
      raise IncompatibleTypes(self, other)

_dtype_to_parakeet_type = {}
def register_scalar_type(ParakeetClass, dtype, equiv_python_types = []):
  parakeet_type = ParakeetClass(dtype)
  _dtype_to_parakeet_type[dtype] = parakeet_type

  python_types = [dtype.type] + equiv_python_types

  for python_type in python_types:
    type_conv.register(python_type, parakeet_type)

  return parakeet_type

class IntT(ScalarT):
  """Base class for bool, signed and unsigned"""

  _members = []

class BoolT(IntT):
  """
  The type is called BoolT to distinguish it from its only instantiation called
  Bool.
  """

  def node_init(self):
    assert dtypes.is_bool(self.dtype)
    self.name = 'bool'

  def __eq__(self, other):
    return other.__class__ is BoolT

Bool = register_scalar_type(BoolT, dtypes.bool8, equiv_python_types = [bool])

class UnsignedT(IntT):
  def node_init(self):
    assert dtypes.is_unsigned(self.dtype), \
        "Expected unsigned but got %s" % self.dtype

UInt8 = register_scalar_type(UnsignedT, dtypes.uint8)

UInt16 = register_scalar_type(UnsignedT, dtypes.uint16)
UInt32 = register_scalar_type(UnsignedT, dtypes.uint32)
UInt64 = register_scalar_type(UnsignedT, dtypes.uint64)

class SignedT(IntT):
  def node_init(self):
    assert dtypes.is_signed(self.dtype)

Int8 = register_scalar_type(SignedT, dtypes.int8)
Int16 = register_scalar_type(SignedT, dtypes.int16)
Int32 = register_scalar_type(SignedT, dtypes.int32)
Int64 = register_scalar_type(SignedT, dtypes.int64,
                             equiv_python_types = [int, long])

class FloatT(ScalarT):
  _members = []
  def node_init(self):
    assert dtypes.is_float(self.dtype)

Float32 = register_scalar_type(FloatT, dtypes.float32)
Float64 = register_scalar_type(FloatT, dtypes.float64,
                               equiv_python_types = [float])

def complex_conj(x):
  np.complex(x.real, -x.imag)

class ComplexT(ScalarT):
  _members = ['elt_type']
  _props_ = [('conjugate', complex_conj)]

  def node_init(self):
    assert dtypes.is_float(self.elt_type), \
        "Expected fields of complex to be floating, got %s" % self.elt_type
    assert dtypes.is_complex(self.dtype)
    self._fields_ = [('real', self.elt_type), ('imag', self.elt_type)]

Complex64 = ComplexT(dtypes.float32, dtypes.complex64)
Complex128 = ComplexT(dtypes.float64, dtypes.complex128)

class ConstIntT(IntT):
  """
  Integer constants get a special type all to their lonesome selves, so we can
  assign types to expressions like "(1,2,3)[0]".
  """

  _members = ['value']

  def __init__(self, value):
    self.dtype = dtypes.int64
    self.value = value

  def combine(self, other):
    if isinstance(other, ConstIntT) and other.value == self.value:
      return self
    else:
      return IntT.combine(self, other)

def is_scalar_subtype(t1, t2):
  return isinstance(t1, ScalarT) and \
         isinstance(t2, ScalarT) and \
         ((t1 == t2) or (t1.nbytes() < t2.nbytes()) or \
          (isinstance(t1, IntT) and isinstance(t2, FloatT)))

def register_numeric_type(klass, dtype):
  parakeet_type = klass(dtype)
  _dtype_to_parakeet_type[dtype] = parakeet_type
  return parakeet_type

def from_dtype(dtype):
  return _dtype_to_parakeet_type[dtype]

def from_char_code(c):
  numpy_type = np.typeDict[c]
  return from_dtype(np.dtype(numpy_type))

def is_scalar(t):
  return isinstance(t, ScalarT)

def all_scalars(ts):
  return all(map(is_scalar, ts))

###########################################
#
#  Pointers!
#
###########################################

class PtrT(ConcreteT):
  """
  I'm not giving pointer a concrete to_python/from_python conversion or any
  usable fields, so it's up to our ctypes_repr and llvm_backend to appropriately
  interpret objects of this type.
  """

  _members = ['elt_type']

  rank = 1
  def index_type(self, idx):
    assert isinstance(idx, IntT), \
        "Index into pointer must be of type int, got %s" % (idx)
    return self.elt_type

  def node_init(self):
    self._ctypes_repr = ctypes.POINTER(self.elt_type.ctypes_repr)

  def __str__(self):
    return "ptr(%s)" % self.elt_type

  def __eq__(self, other):
    return isinstance(other, PtrT) and self.elt_type == other.elt_type

  def __hash__(self):
    return hash(self.elt_type)

  def __repr__(self):
    return str(self)

  @property
  def ctypes_repr(self):
    return self._ctypes_repr

_ptr_types = {}
def ptr_type(t):
  if t in _ptr_types:
    return _ptr_types[t]
  else:
    ptr_t = PtrT(t)
    _ptr_types[t] = ptr_t
    return ptr_t
