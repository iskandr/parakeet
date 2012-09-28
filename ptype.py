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

  def __str__(self):
    return repr(self)
  
  
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
  _members = ['dtype']
  
  def __init__(self, dtype, name = None):
    if not isinstance(dtype, np.dtype):
      dtype = np.dtype(dtype)
    
    self.dtype = dtype 
    
    if name is None:
      name = dtype.type.__name__
    self.name = name   
  
  def __eq__(self, other):
    return isinstance(other, Scalar) and other.dtype == self.dtype 
 
  def __hash__(self):
    return hash(self.dtype)
  
  def __repr__(self):
    return self.name 


  def is_float(self):
    return self.dtype.type in np.sctypes['float']
  
  def is_signed(self):
    return self.dtype.type in np.sctypes['int']
  
  def is_unsigned(self):
    return self.dtype.type in np.sctypes['uint']
  
  def is_bool(self):
    return self.dtype == np.bool8 
  
  def is_int(self):
    return self.is_bool() or self.is_signed() or self.is_unsigned()
  
  def nbytes(self):
    return self.dtype.itemsize

  def combine(self, other):
    if isinstance(other, Scalar):
      combined_dtype = np.promote_types(self.dtype, other.dtype)
      if combined_dtype == self.dtype:
        return self
      elif combined_dtype == other.dtype:
        return other
      else:
        return Scalar(combined_dtype)
      
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
      raise IncompatibleTypes(self, other)
    
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
  
  def combine(self, other):
    if isinstance(other, Tuple) and len(other.elt_types) == len(self.elt_types):
      combined_elt_types = [t1.combine(t2) for \
                            (t1, t2) in zip(self.elt_types, other.elt_tyepes)]
      if combined_elt_types != self.elt_types:
        return Tuple(combined_elt_types)
      else:
        return self
    else:
      raise IncompatibleTypes(self, other)

class Closure:
  def __init__(self, fn, args):
    self.fn = fn
    self.args = args 
    
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
  


_dtype_to_parakeet = {}
_parakeet_to_dtype = {}
def register_scalar_type(dtype, name = None):
  if not isinstance(dtype, np.dtype):
    dtype = np.dtype(dtype)
  parakeet_type = Scalar(dtype, name)
  _dtype_to_parakeet[dtype] = parakeet_type
  _parakeet_to_dtype[parakeet_type] = dtype
  return parakeet_type

def from_dtype (dtype):
  return _dtype_to_parakeet[dtype] 

Bool = register_scalar_type(np.bool8, 'bool')
Int8 = register_scalar_type(np.int8)
Int16 = register_scalar_type(np.int16)
Int32 = register_scalar_type(np.int32)
Int64 = register_scalar_type(np.int64)
Float32 = register_scalar_type(np.float32)
Float64 = register_scalar_type(np.float64)



  
def type_of_scalar(x):
  assert np.isscalar(x)
  if isinstance(x, int):
      x = np.int64(x)
  elif isinstance(x, float):
    x = np.float64(x)
  else:
    assert hasattr(x, 'dtype')
  return from_dtype(x.dtype)

def type_of_value(x):
  if np.isscalar(x):
    return type_of_scalar(x)
  elif isinstance(x, tuple):
    elt_types = map(type_of_value, x)
    return Tuple(elt_types)
  elif isinstance(x, np.ndarray):
    return Array(from_dtype(x.dtype), np.rank(x))
  else:
    raise RuntimeError("Unsupported type " + str(type(x)))
  
def combine_type_list(types):
  common_type = Unknown 

  for t in types:
    common_type = common_type.combine(t)
  return common_type

