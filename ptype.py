import numpy as np 
import numpy_type_info 

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

class Type(TreeLike):  
  def nbytes(self):
    raise RuntimeError("nbytes not implemented")

  def combine(self, other):
    raise IncompatibleTypes(self, other)
  

class Any(Type):
  """top of the type lattice, absorbs all types"""
  _members = []
  def combine(self, other):
    return self
  
# since there's only one Any type, just create an instance of the same name
Any = Any()

class Unknown(Type):
  """Bottom of the type lattice, absorbed by all other types"""
  _members = []
  def  combine(self, other):
    return other

#single instance of the Unknown type with same name
Unknown = Unknown()

## look up types by their number of bytes
#def find_float_dtype_by_nbytes(nbytes):
#  if nbytes <= 4:
#    return np.float32
#  else:
#    return np.float64

  


# base class for all concrete scalar types
# don't actually tag any values with this
class Scalar(Type):
  rank = 0
  #_members = ['dtype']
  
  def __init__(self, dtype, name = None):
    self.dtype = dtype 
    
    if name is None:
      name = dtype.__name__
    self.name = name   
  
  def __eq__(self, other):
    return isinstance(other, Scalar) and other.dtype == self.dtype 
 
  def __hash__(self):
    return hash(self.dtype)
  
  def __repr__(self):
    return self.name 

  def is_float(self):
    return self.dtype in [np.float32, np.float64]

  def is_signed(self):
    return self.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]
  
  def is_int(self):
    return not self.is_float()

  def nbytes(self):
    return numpy_type_info.byte_sizes[self.dtype]

  def combine(self, other):
    if isinstance(other, Scalar):
      combined_type = numpy_type_info.combine(self.dtype, other.dtype)
      if combined_type:
        return combined_type
      else:
        raise IncompatibleTypes(self, other)
    elif isinstance(other, Array):
      raise RuntimeError("Array not implemented")
    else:
      raise IncompatibleTypes(self, other)
      
def is_scalar_subtype(t1, t2):
  return isinstance(t1, Scalar) and \
    isinstance(t2, Scalar) and \
    ((t1 == t2) or (t1.nbytes() < t2.nbytes()) or (t1.is_int() and t2.is_float()))


class CompoundType(Type):
  pass 

class Array(CompoundType):
  _members = ['elt_type', 'rank']
  
  def __init__(self, elt_type, rank):
    assert isinstance(elt_type, Scalar)
    CompoundType.__init__(elt_type, rank)

  def nbytes(self):
    raise RuntimeError("Can't get size of an array just from its type")  

  def dtype(self):
    return self.elt_type.dtype()
 
  def __repr__(self):
    return "array(%s, %d)" % (self.elt_type, self.rank)

  def __eq__(self, other): 
    return isinstance(other, Array) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    else:
      raise TypeFailure()
class Tuple(CompoundType):
  rank = 0 
  _members = ['elt_types']
  
  def nbytes(self):
    return sum([t.nbytes() for t in self.elt_types])
  
  def dtype(self):
    raise RuntimeError("Do tuples have dtypes?")
  
  def __eq__(self, other):
    return isinstance(other, Tuple) and self.elt_types == other.elt_types 

  def __hash__(self):
    return hash(self.elt_types)



class Closure(Type): 
  """
  Closures statically refer to the untyped function id they close over 
  """
  _members = ['fn', 'args']
  

class ClosureSet(Type):
  """
  If multiple closures meet along control flow paths then join them into a closure set
  """
  _members = ['closures'] 
  
  def __init__(self, *closure_types):
    self.closures = set(closure_types)
  
  


# preallocate all the scalar types
# as an optimiztion so we don't 
# end up allocating lots of identical
# objects 
Bool = Scalar(np.bool8, 'bool')
Int8 = Scalar(np.int8)
Int16 = Scalar(np.int16)
Int32 = Scalar(np.int32)
Int64 = Scalar(np.int64)
Float32 = Scalar(np.float32)
Float64 = Scalar(np.float64)

_dtype_to_type = { 
  np.bool8 : Bool, 
  np.int8 : Int8, 
  np.int16 : Int16, 
  np.int32 : Int32, 
  np.int64 : Int64, 
  np.float32 : Float32, 
  np.float64 : Float64
}

def dtype_to_type(dtype):
  return _dtype_to_type[dtype] 
  
def type_of_scalar(x):
  assert np.isscalar(x)
  if isinstance(x, int):
    x = np.int64(x)
  elif isinstance(x, float):
    x = np.float64(x)
  else:
    assert hasattr(x, 'dtype')
  return dtype_to_type(x.dtype)

def type_of_value(x):
  if np.isscalar(x):
    return type_of_scalar(x)
  elif isinstance(x, tuple):
    elt_types = map(type_of_value, x)
    return Tuple(elt_types)
  elif isinstance(x, np.ndarray):
    return Array(dtype_to_type(x.dtype), np.rank(x))
  else:
    raise RuntimeError("Unsupported type " + str(type(x)))
