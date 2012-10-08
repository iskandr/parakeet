import numpy as np 
import ctypes 
import abc 
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
  

class AnyT(Type):
  """top of the type lattice, absorbs all types"""

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



class ConcreteT(Type):
  
  __metaclass__ = abc.ABCMeta 
  
  
  @abc.abstractmethod 
  def from_python(self, x):
    """
    Given a python object, convert it to
    the internal representation of this type.
    """ 
    assert False, "from_python not implemented for %s" % self.node_type()
  
   
  @abc.abstractmethod
  def to_python(self, x):
    """
    Create an object which be used by python code outside
    parakeet
    """
    assert False, "to_python not implemented for %s" % self.node_type()

  @property
  def nbytes(self):
    return ctypes.sizeof(self.ctypes_repr)

  
  def to_ctypes(self, x):
    """
    Given a runtime value which has been converted to the internal representation
    for this type (via the from_python method), create a ctypes object
    which can be later be passed to native code
    """
     
  

# base class for all concrete scalar types
# don't actually tag any values with this
class ScalarT(ConcreteT):
  rank = 0
  _members = ['dtype']

  def finalize_init(self):
    assert isinstance(self.dtype, np.dtype)
    self.name = self.dtype.name
    
    ctypes_name = 'c_' + self.name
    if hasattr(ctypes, ctypes_name):
      self.internal_repr = getattr(ctypes, ctypes_name)
    elif self.name == 'float16' or self.name == 'float32':
      self.internal_repr = ctypes.c_float
    elif self.name == 'float64':
      self.internal_repr = ctypes.c_double
    else:
      raise RuntimeError("Can't map dtype %s to ctypes representation" % self.name) 
    
      
  def __eq__(self, other):
    return isinstance(other, ScalarT) and other.dtype == self.dtype 
 
  def __hash__(self):
    return hash(self.dtype)
  
  def __repr__(self):
    return self.name 
  
  def __str__(self):
    return str(self.name)

  
  def from_python(self, x):
    return self.dtype.type(x)
  
  def to_python(self, x):
    """
    Since we use numpy's scalars for our internal representation
    we can just give them back to Python without a problem
    """
    return x  

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
  def finalize_init(self):
    assert dtype_is_float(self.dtype)
    self.name = self.dtype.type.__name__

class IntT(ScalarT):
  """Base class for bool, signed and unsigned"""
  pass

  
class BoolT(IntT):
  """The type is called BoolT to distinguish it from its only instantiation called Bool."""
  def finalize_init(self):
    assert dtype_is_bool(self.dtype)
    self.name = 'bool'
    
class SignedT(IntT):
  def finalize_init(self):
    assert dtype_is_signed(self.dtype)
     

class UnsignedT(IntT):
  def finalize_init(self):
    assert dtype_is_unsigned(self.dtype)
    
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


def buffer_info(buf, ptr_type = ctypes.c_void_p):
  assert isinstance(buf, buffer)
  """Given a python buffer, return its address and length"""
  address = ptr_type()
  length = ctypes.c_ssize_t() 
  obj =  ctypes.py_object(buf) 
  ctypes.pythonapi.PyObject_AsReadBuffer(obj, ctypes.byref(address), ctypes.byref(length))
  return address, length   
  
  
class BufferT(ConcreteT):
  """
  Wrap a python buffer, which combines a pointer and its data size
  """ 
  _members = ['elt_type']

  def finalize_init(self):
    self.ctypes_pointer_t = ctypes.POINTER(self.elt_type.internal_repr)
    class BufferRepr(ctypes.Structure):
      _fields_ = [
        ('pointer', self.ctypes_pointer_t),
        ('length', ctypes.c_int64)
      ]   
    self.internal_repr = BufferRepr  

  
  def from_python(self, x):
    assert isinstance(x, buffer)
    ptr, length = buffer_info(x, self.ctypes_pointer_t)
    return self.internal_repr(ptr, length)
  
  def to_python(self, x):
    """
    For now, to avoid to dealing with the messiness of ownership,  
    we just always copy data on the way out of Parakeet
    """
    dest_buf = ctypes.pythonapi.PyBuffer_New(x.length)
    dest_ptr, _ = buffer_info(dest_buf, self.ctypes_pointer_t)
    
    # copy data from pointer
    ctypes.memmove(dest_ptr, x.pointer, x.length)
    return dest_buf 
    

_buffer_types = {}
def make_buffer_type(t):
  """Memoizing constructor for pointer types"""
  if t in _buffer_types:
    return _buffer_types[t]
  else:
    b = BufferT(t)
    _buffer_types[t] = b
    return b


def ctypes_struct_from_fields(field_types):
  ctypes_fields = []
    
  for (field_name, parakeet_type) in field_types:
    ctypes_fields.append( (field_name, parakeet_type.internal_repr) )

  class InternalRepr(ctypes.Structure):
    _fields_ = ctypes_fields
  return InternalRepr 


class StructT(ConcreteT):
  """
  Structs must define how to translate themselves
  from their normal python representation to a simplified 
  form involving only base types. 
   
  Any derived class *must* define a _field_types property
  which contains a list of (name, field_type) pairs.
  """
  pass


class TupleT(StructT):
  rank = 0 
  _members = ['elt_types']
  
  def finalize_init(self):
    self._field_types = [ 
      ("elt%d" % i, t) for (i,t) in enumerate(self.elt_types)
    ]
    self.internal_repr = ctypes_struct_from_fields(self._field_types)                           

  def from_python(self, python_tuple):
    assert isinstance(python_tuple, tuple)
    assert len(python_tuple) == len(self.elt_types)
    
    converted_elts = []
    for (elt_type, elt_value) in zip(self.elt_types, python_tuple):
      converted_elts.append( elt_type.from_python(elt_value) )
      
    assert isinstance(self.internal_repr, ctypes.Structure)
    assert len(self.internal_repr._fields_) == len(converted_elts)
    return self.internal_repr(*converted_elts)
  
  def to_python(self, struct):
    python_elt_values = []
    for (i, elt_t) in enumerate(self.elt_types):
      field_name = "elt%d" % i
      internal_field_value = getattr(struct, field_name)
      python_elt_value = elt_t.to_python(internal_field_value)
      python_elt_values.append(python_elt_value)
    return tuple(python_elt_values)
                             
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
  
_tuple_types = {}
def repeat_tuple(t, n):
  """Given the base type t, construct the n-tuple t*t*...*t"""
  elt_types = tuple([t] * n)
  if elt_types in _tuple_types:
    return _tuple_types[elt_types]
  else:
    tuple_t = TupleT(elt_types)
    _tuple_types[elt_types] = tuple_t
    return tuple_t 


def make_tuple_type(elt_types):
  """
  Use this memoized construct to avoid
  constructing too many distinct tuple type
  objects and speeding up equality checks
  """ 
  key = tuple(elt_types)
  if key in _tuple_types:
    return _tuple_types[key]
  else:
    t = TupleT(key)
    _tuple_types[key] = t
    return t 


class ArrayT(StructT):
  _members = ['elt_type', 'rank']

  def finalize_init(self):
    assert isinstance(self.elt_type, ScalarT)
    tuple_t = repeat_tuple(Int64, self.rank)
    
    self._field_types = [
      ('data', make_buffer_type(self.elt_type)), 
      ('shape', tuple_t), 
      ('strides', tuple_t),
    ]
    self.internal_repr = ctypes_struct_from_fields(self._field_types)  

  def from_python(self, x):
    return self.internal_repr()
  
  def to_python(self, x):
    """
    WARNING! 
    If a pointer gets passed into and back out of 
    parakeet we're going to screw up the reference counts
    on its buffer! 
    """
    assert False, "Haven't yet implemented construction of buffers"
    
  def nbytes(self):
    raise RuntimeError("Can't get size of an array just from its type")  

  def dtype(self):
    return self.elt_type.dtype()
 
  def __eq__(self, other): 
    return isinstance(other, ArrayT) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)

    

class ClosureT(StructT):
  _members = ['fn', 'args']
  
  def finalize_init(self):
    if self.args is None:
      self.args = ()
    elif not hasattr(self.args, '__iter__'):
      self.args = tuple([self.args])
    elif not isinstance(self.args, tuple):
      self.args = tuple(self.args)
      
    self._fields_types = [('fn_id', Int64)] 
    for (i, t) in enumerate(self.args):
      self._field_types.append( ('arg%d' % i, t.internal_repr) )
      
    self.internal_repr = ctypes_struct_from_fields(self._fields_types) 

  def from_python(self, fn_val):
    assert False, "Not yet sure how to translate closure types from python"
  
  def to_python(self, clos):
    assert False, "Not yet sure how to return closures back to python"
    
    
  def __hash__(self):
    return hash(repr(self))
  
  def __eq__(self, other):
    return self.fn == other.fn and self.args == other.args
  
  def combine(self, other):
    if isinstance(other, ClosureSet):
      return other.combine(self)
    elif isinstance(other, ClosureT):
      if self == other:
        return self
      else:
        return ClosureSet(self, other)
    else:
      raise IncompatibleTypes(self, other)
    
class ClosureSet(Type):
  """
  If multiple closures meet along control flow paths then join them into a closure set.
  This type should not appear by the time we're generating LLVM code.
  """
  _members = ['closures'] 
  
  def __init__(self, *closures):
    self.closures = set([])
    for clos_t in closures:
      if isinstance(clos_t, ClosureSet):
        self.closures.update(clos_t.closures)
      else:
        assert isinstance(clos_t, ClosureT)
        self.closures.add(clos_t)

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
  
  def __iter__(self):
    return iter(self.closures)
  
  def __len__(self):
    return len(self.closures)

  
def type_of_scalar(x):
  if isinstance(x, bool):
    return Bool 
  elif isinstance(x, (int, long)):
    return Int64
  elif isinstance(x, float):
    return Float64
  else:
    raise RuntimeError("Unsupported scalar type: %s" % x)
  
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

