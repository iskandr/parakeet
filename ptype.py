import numpy as np 

from tree import TreeLike

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
  
  
class Type(TreeLike):  
  def nbytes(self):
    raise RuntimeError("nbytes not implemented")

  def combine(self, other):
    raise IncompatibleTypes(self, other)
  

class AnyT(Type):
  """top of the type lattice, absorbs all types"""
  _members = []
  def combine(self, other):
    return self
  
# since there's only one Any type, just create an instance of the same name
Any = AnyT()

class UnknownT(Type):
  """Bottom of the type lattice, absorbed by all other types"""
  _members = []
  def  combine(self, other):
    return other

#single instance of the Unknown type with same name
Unknown = UnknownT()

## look up types by their number of bytes
#def find_float_dtype_by_nbytes(nbytes):
#  if nbytes <= 4:
#    return np.float32
#  else:
#    return np.float64

  


# base class for all concrete scalar types
# don't actually tag any values with this
class ScalarT(Type):
  rank = 0
  _members = ['dtype']
  
  def __init__(self, dtype, name = None):
    assert False, "Should not be called directly"
    
  def __eq__(self, other):
    return isinstance(other, ScalarT) and other.dtype == self.dtype 
 
  def __hash__(self):
    return hash(self.dtype)
  
  def __repr__(self):
    return self.name 
  
  def __str__(self):
    return str(self.name)

  def nbytes(self):
    return self.dtype.itemsize

  def combine(self, other):
    if isinstance(other, ScalarT):
      combined_dtype = np.promote_types(self.dtype, other.dtype)
      if combined_dtype == self.dtype:
        return self
      elif combined_dtype == other.dtype:
        return other
      else:
        return scalar_type_from_dtype(combined_dtype)
      
    elif isinstance(other, ArrayT):
      raise RuntimeError("Array not implemented")
    else:
      raise IncompatibleTypes(self, other)
      



def dtype_is_float(dtype):
  return dtype.type in np.sctypes['float']

def dtype_is_signed(dtype):
  return dtype.type in np.sctypes['int']
  
def dtype_is_unsigned(dtype):
  return dtype.type in np.sctypes['uint'] 
  
def dtype_is_bool(dtype):
  return dtype == np.bool8 

def is_int(dtype):
  return dtype_is_bool(dtype) or dtype_is_signed(dtype) or dtype_is_unsigned(dtype)
  
class FloatT(ScalarT):
  def __init__(self, dt):
    assert dtype_is_float(dt)
    self.dtype = dt 
    self.name = dt.type.__name__

class IntT(ScalarT):
  """Base class for bool, signed and unsigned"""
  pass

  
class BoolT(IntT):
  """The type is called BoolT to distinguish it from its only instantiation called Bool."""
  def __init__(self, dt):
    assert dtype_is_bool(dt)
    self.dtype = dt
    self.name = 'bool'

    
class SignedT(IntT):
  def __init__(self, dt):
    assert dtype_is_signed(dt)
    self.dtype = dt  
    self.name = dt.type.__name__

class UnsignedT(IntT):
  def __init__(self, dt):
    assert dtype_is_unsigned(dt)
    self.dtype = dt  
    self.name = dt.type.__name__

_dtype_to_parakeet = {}
_parakeet_to_dtype = {}

def scalar_type_from_dtype(dtype):
  if dtype_is_float(dtype):
    return FloatT(dtype)
  elif dtype_is_signed(dtype):
    return SignedT(dtype)
  elif dtype_is_unsigned(dtype):
    return UnsignedT(dtype)
  elif dtype_is_bool(dtype):
    return BoolT(dtype)
  else:
    assert False, "Don't know how to register type: %s" % dtype 
  

def register_scalar_type(dtype):
  if not isinstance(dtype, np.dtype):
    dtype = np.dtype(dtype)
  parakeet_type = scalar_type_from_dtype(dtype)
  _dtype_to_parakeet[dtype] = parakeet_type
  _parakeet_to_dtype[parakeet_type] = dtype
  return parakeet_type

def from_dtype (dtype):
  return _dtype_to_parakeet[dtype] 

def from_char_code(c):
  numpy_type = np.typeDict[c]
  return from_dtype(np.dtype(numpy_type))

Bool = register_scalar_type(np.bool8)

UInt8 = register_scalar_type(np.uint8)
UInt16 = register_scalar_type(np.uint16)
UInt32 = register_scalar_type(np.uint32)
UInt64 = register_scalar_type(np.uint64)

Int8 = register_scalar_type(np.int8)
Int16 = register_scalar_type(np.int16)
Int32 = register_scalar_type(np.int32)
Int64 = register_scalar_type(np.int64)

Float16 = register_scalar_type(np.float16)
Float32 = register_scalar_type(np.float32)
Float64 = register_scalar_type(np.float64)

def is_scalar_subtype(t1, t2):
  return isinstance(t1, ScalarT) and \
    isinstance(t2, ScalarT) and \
    ((t1 == t2) or (t1.nbytes() < t2.nbytes()) or \
     (isinstance(t1, IntT) and isinstance(t2, FloatT)))



class CompoundType(Type):
  pass 

class ArrayT(CompoundType):
  _members = ['elt_type', 'rank']
  
  def __init__(self, elt_type, rank):
    assert isinstance(elt_type, ScalarT)
    CompoundType.__init__(elt_type, rank)

  def nbytes(self):
    raise RuntimeError("Can't get size of an array just from its type")  

  def dtype(self):
    return self.elt_type.dtype()
 
  def __repr__(self):
    return "array(%s, %d)" % (self.elt_type, self.rank)

  def __eq__(self, other): 
    return isinstance(other, ArrayT) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)
    
class TupleT(CompoundType):
  rank = 0 
  _members = ['elt_types']
  
  def nbytes(self):
    return sum([t.nbytes() for t in self.elt_types])
  
  def dtype(self):
    raise RuntimeError("Do tuples have dtypes?")
  
  def __eq__(self, other):
    return isinstance(other, TupleT) and self.elt_types == other.elt_types 

  def __hash__(self):
    return hash(self.elt_types)
  
  def combine(self, other):
    if isinstance(other, TupleT) and len(other.elt_types) == len(self.elt_types):
      combined_elt_types = [t1.combine(t2) for \
                            (t1, t2) in zip(self.elt_types, other.elt_tyepes)]
      if combined_elt_types != self.elt_types:
        return TupleT(combined_elt_types)
      else:
        return self
    else:
      raise IncompatibleTypes(self, other)

class ClosureSig:
  def __init__(self, fn, args = ()):
    self.fn = fn
    self.args = tuple(args) 
    
  def __repr__(self):
    return "Closure(fn=%s, args=%s)" % (self.fn, self.args)
  
  def __hash__(self):
    return hash(repr(self))
  
  def __eq__(self, other):
    return self.fn == other.fn and self.args == other.args
    
class ClosureSet(Type):
  """
  If multiple closures meet along control flow paths then join them into a closure set
  """
  _members = ['closures'] 
  
  def __init__(self, *closure_types):
    self.closures = set(closure_types)
  
  def combine(self, other):
    if isinstance(other, ClosureSet):
      combined_closures = self.closures.union(other.closures)
      if combined_closures != self.closures:
        return ClosureSet(combined_closures)
      else:
        return self 
    else:
      raise IncompatibleTypes(self, other)
  
  def __eq__(self, other):
    return self.closures == other.closures 


  
def type_of_scalar(x):
  return from_dtype(np.min_scalar_type(x))
 
def type_of_value(x):
  if np.isscalar(x):
    return type_of_scalar(x)
  elif isinstance(x, tuple):
    elt_types = map(type_of_value, x)
    return TupleT(elt_types)
  elif isinstance(x, np.ndarray):
    return ArrayT(from_dtype(x.dtype), np.rank(x))
  else:
    raise RuntimeError("Unsupported type " + str(type(x)))
  
def combine_type_list(types):
  common_type = Unknown 

  for t in types:
    common_type = common_type.combine(t)
  return common_type

