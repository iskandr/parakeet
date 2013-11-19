import numpy as np

import dtypes 
import type_conv 

from core_types import IncompatibleTypes, ImmutableT

# base class for all concrete scalar types
# don't actually tag any values with this
class ScalarT(ImmutableT):
  rank = 0
  _members = ['dtype']

  def node_init(self):
    assert isinstance(self.dtype, np.dtype), \
        "Expected dtype, got %s" % self.dtype
    self.name = self.dtype.name
    self.nbytes = self.dtype.itemsize

  @property
  def ctypes_repr(self):
    return dtypes.to_ctypes(self.dtype)

  def __eq__(self, other):
    if self is other:
      return True 
    if self.__class__ is not other.__class__:
      return False 
    return self.nbytes == other.nbytes
  
  def __ne__(self, other):
    return not (self == other) 
  
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

class Int24T(SignedT):
  """
  We don't actually support 24-bit integers, but they're useful  
  for implemented corner-case casting logic, such as summing booleans
  """
  def __init__(self):
    self.name = "int24"
    self.nbytes = 3 
    
  def combine(self, other):
    if isinstance(other, IntT):
      if other.nbytes <= self.nbytes:
        return Int32 
    return other 
  
  def __hash__(self):
    return hash("int24")

  def __eq__(self, other):
    return self.__class__ is other.__class__ 
    
  def __ne__(self, other):
    return self.__class__ is not other.__class__
  
Int24 = Int24T()

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
         ((t1 == t2) or (t1.nbytes < t2.nbytes) or \
          (isinstance(t1, IntT) and isinstance(t2, FloatT)))

def from_dtype(dtype):
  if not isinstance(dtype, np.dtype):
    assert hasattr(dtype, 'dtype'), "Expected a dtype but %s" % dtype 
    value = dtype(0)
    dtype = value.dtype 
  assert dtype in _dtype_to_parakeet_type, \
    "Don't know how to make Parakeet scalar type from dtype %s" % dtype
  return _dtype_to_parakeet_type[dtype]

def from_char_code(c):
  numpy_type = np.typeDict[c]
  return from_dtype(np.dtype(numpy_type))

def is_scalar(t):
  return isinstance(t, ScalarT)

def all_scalars(ts):
  return all(map(is_scalar, ts))
