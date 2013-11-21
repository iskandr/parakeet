import numpy as np

import dtypes 
import type_conv 

from core_types import IncompatibleTypes, ImmutableT

# base class for all concrete scalar types
# don't actually tag any values with this
class ScalarT(ImmutableT):
  rank = 0
  
  def __init__(self, dtype):
    self.dtype = dtype 
    self.name = self.dtype.name
    self.nbytes = self.dtype.itemsize
    self._hash = hash(self.dtype)
  
  def __eq__(self, other):
    return (self is other) or ( self.__class__ is other.__class__ and self.nbytes == other.nbytes)
  
  def __ne__(self, other):
    return not (self == other) 
  
  def __hash__(self):
    return self._hash

  def __repr__(self):
    return self.name

  def __str__(self):
    return str(self.name)

  def index_type(self, _):
    return self

  def convert_python_value(self, x):
    return self.dtype.type(x)
  
  def combine(self, other):
    if hasattr(other, 'dtype'):
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
  pass 

  

class BoolT(IntT):
  """
  The type is called BoolT to distinguish it from its only instantiation called
  Bool.
  """
  
  def __init__(self, dtype):
    ScalarT.__init__(self, dtype)
    self.name = 'bool'

  def __eq__(self, other):
    return other.__class__ is BoolT

Bool = register_scalar_type(BoolT, dtypes.bool8, equiv_python_types = [bool])

class UnsignedT(IntT):
  pass 

UInt8 = register_scalar_type(UnsignedT, dtypes.uint8)

UInt16 = register_scalar_type(UnsignedT, dtypes.uint16)
UInt32 = register_scalar_type(UnsignedT, dtypes.uint32)
UInt64 = register_scalar_type(UnsignedT, dtypes.uint64)

class SignedT(IntT):
  pass 

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
    self._hash = hash("int24")
    
  def combine(self, other):
    if isinstance(other, IntT):
      if other.nbytes <= self.nbytes:
        return Int32 
    return other 
  
  def __hash__(self):
    return self._hash 
  
  def __eq__(self, other):
    return self.__class__ is other.__class__ 
    
  def __ne__(self, other):
    return self.__class__ is not other.__class__
  
Int24 = Int24T()

class FloatT(ScalarT):
  pass 

Float32 = register_scalar_type(FloatT, dtypes.float32)
Float64 = register_scalar_type(FloatT, dtypes.float64,
                               equiv_python_types = [float])

def is_scalar_subtype(t1, t2):
  return isinstance(t1, ScalarT) and \
         isinstance(t2, ScalarT) and \
         ((t1 is t2) or 
          (t1 == t2) or 
          (t1.nbytes < t2.nbytes) or 
          (isinstance(t1, IntT) and isinstance(t2, FloatT)))

def from_dtype(dtype):
  if not isinstance(dtype, np.dtype):
    assert hasattr(dtype, 'dtype'), "Expected a dtype but %s" % dtype 
    value = dtype(0)
    dtype = value.dtype 
  parakeet_type = _dtype_to_parakeet_type.get(dtype)
  if parakeet_type is None:
    assert False, "Don't know how to make Parakeet scalar type from dtype %s" % dtype
  return parakeet_type

def from_char_code(c):
  numpy_type = np.typeDict[c]
  return from_dtype(np.dtype(numpy_type))

def is_scalar(t):
  return isinstance(t, ScalarT)

def all_scalars(ts):
  return all(is_scalar(t) for t in  ts)
