import numpy as np 
from tree import TreeLike

class Type(TreeLike):  
  def nbytes(self):
    raise RuntimeError("nbytes not implemented")
    

byte_sizes = { 
  np.int8 : 1, 
  np.int16 : 2, 
  np.int32 : 4, 
  np.int64 : 8, 
  np.float32 : 4, 
  np.float64 : 8, 
  np.uint8 : 1, 
  np.uint16 : 2, 
  np.uint32 : 4, 
  np.uint64 : 8
}

# base class for all concrete scalar types
# don't actually tag any values with this
class Scalar(Type):
  rank = 0
  _members = ['dtype']
  
  def __eq__(self, other):
    return isinstance(other, Scalar) and other.dtype == self.dtype 

  def __repr__(self):
    return str(self.dtype)

  def is_float(self):
    return self.dtype in [np.float32, np.float64]

  def is_int(self):
    return not self.is_float()

  def nbytes(self):
    return byte_sizes[self.dtype]

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

class Tuple(CompoundType):
  rank = 0 
  _members = ['elt_types']
  
  def nbytes(self):
    return sum([t.nbytes() for t in self.elt_types])
  
  def dtype(self):
    raise RuntimeError("Do tuples have dtypes?")
  
  def __eq__(self, other):
    return isinstance(other, Tuple) and self.elt_types == other.elt_types 

def is_scalar_subtype(t1, t2):
  return isinstance(t1, Scalar) and \
    isinstance(t2, Scalar) and \
    ((t1 == t2) or (t1.nbytes() < t2.nbytes()) or (t1.is_int() and t2.is_float()))

# preallocate all the scalar types
# as an optimiztion so we don't 
# end up allocating lots of identical
# objects 
Int8 = Scalar(np.int8)
Int16 = Scalar(np.int16)
Int32 = Scalar(np.int32)
Int64 = Scalar(np.int64)
Float32 = Scalar(np.float32)
Float64 = Scalar(np.float64)

_dtype_to_type = { 
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
