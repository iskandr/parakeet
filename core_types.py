from node import Node 
import numpy as np 
import type_conv
import dtypes 


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


class NoneT(Type):
  _members = []
  def combine(self, other):
    if isinstance(other, NoneT):
      return self
    else:
      raise IncompatibleTypes(self, other)
  
  

NoneType = NoneT()

def combine_type_list(types):
  common_type = Unknown 

  for t in types:
    common_type = common_type.combine(t)
  return common_type


import abc 

class ConcreteT(Type):
  __meta__ = abc.ABCMeta
  
  """
  Type which actually have some corresponding runtime values, 
  as opposed to "Any" and "Unknown"
  """
  
  @abc.abstractproperty
  def ctypes_repr(self):
    pass
  
  def from_python(self, py_val):
    return self.ctypes_repr(py_val)
  
  def to_python(self, internal):
    return internal
  
import ctypes 

def is_struct(c_repr):
  return type(c_repr) == type(ctypes.Structure)


class StructT(Type):
  """
  All concrete types excluding scalars and pointers
  """

  # expect each child class to fill this list 
  _fields_ = []
  
  _repr_cache = {}
  
  
  def field_type(self, name):
    for (field_name, field_type) in self._fields_:
      if field_name == name:
        return field_type
    raise RuntimeError("Couldn't find field named '%s' for %s" % (name, self))
  
  def field_pos(self, name):
    for (i, (field_name, _)) in enumerate(self._fields_):
      if field_name == name:
        return i
    raise RuntimeError("Couldn't find field named '%s' for %s" % (name, self))
  
  @property
  def ctypes_repr(self):
    if self in self._repr_cache:
      return self._repr_cache[self]
  
    ctypes_fields = []
    
    for (field_name, parakeet_field_type) in self._fields_:
      
      field_repr = parakeet_field_type.ctypes_repr
      # nested structures will be heap allocated 
      if isinstance(parakeet_field_type,  StructT):
        # print "POINTER", field_name, field_repr
        ptr_t = ctypes.POINTER(field_repr) 
        ctypes_fields.append( (field_name, ptr_t) )
      else:
        # print "NOT A POINTER", field_name, field_repr 
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
# helper functions to implement properties of 
# python scalar objects
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
    assert isinstance(self.dtype, np.dtype)
    self.name = self.dtype.name

  @property
  def ctypes_repr(self):
    return dtypes.to_ctypes(self.dtype)      

  @property
  def nbytes(self):
    return self.dtype.itemsize 
  
  def __eq__(self, other):
    return isinstance(other, ScalarT) and other.dtype == self.dtype 
 
  def __hash__(self):
    return hash(self.dtype)
  
  def __repr__(self):
    return self.name 
  
  def __str__(self):
    return str(self.name)

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
  # print "REGISTERING", dtype,  _dtype_to_parakeet_type
  
  python_types = [dtype.type] + equiv_python_types 
  
  for python_type in python_types:
      type_conv.register(python_type, parakeet_type) 

  return parakeet_type 
   
class IntT(ScalarT):
  """Base class for bool, signed and unsigned"""
  pass

  
class BoolT(IntT):
  """The type is called BoolT to distinguish it from its only instantiation called Bool."""
  def node_init(self):
    assert dtypes.is_bool(self.dtype)
    self.name = 'bool'
    
Bool = register_scalar_type(BoolT, dtypes.bool8, equiv_python_types = [bool])

class UnsignedT(IntT):
  def node_init(self):
    assert dtypes.is_unsigned(self.dtype), "Expected unsigned but got %s" % self.dtype


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
Int64 = register_scalar_type(SignedT, dtypes.int64, equiv_python_types = [int, long])


class FloatT(ScalarT):
  def node_init(self):
    assert dtypes.is_float(self.dtype)

Float32 = register_scalar_type(FloatT, dtypes.float32)
Float64 = register_scalar_type(FloatT, dtypes.float64, equiv_python_types = [float])



def complex_conj(x):
  np.complex(x.real, -x.imag)

class ComplexT(ScalarT):
  _members = ['elt_type', 'dtype']
  _props_ = [('conjugate', complex_conj)]
  
  def node_init(self):
    assert dtypes.is_float(self.elt_type)
    assert dtypes.is_complex(self.dtype)
    self._fields_ = [('real', self.elt_type), ('imag', self.elt_type)]
      
Complex64 = ComplexT(dtypes.float32, dtypes.complex64)
Complex128 = ComplexT(dtypes.float64, dtypes.complex128)


class ConstIntT(IntT):
  """
  Integer constants get a special type all to their lonesome selves, 
  so we can assign types to expressions like "(1,2,3)[0]". 
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
  # print "Looking up", dtype 
  # print "Current table:", _dtype_to_parakeet_type
  return _dtype_to_parakeet_type[dtype]

def from_char_code(c):
  numpy_type = np.typeDict[c]
  return from_dtype(np.dtype(numpy_type))



###########################################
#
#  Closures! 
#
###########################################


class ClosureT(StructT):
  _members = ['fn', 'args']
  
  def node_init(self):
    if self.args is None:
      self.args = ()
    elif not hasattr(self.args, '__iter__'):
      self.args = tuple([self.args])
    elif not isinstance(self.args, tuple):
      self.args = tuple(self.args)
      
    self._fields_ = [('fn_id', Int64)] 
    for (i, t) in enumerate(self.args):
      self._fields_.append( ('arg%d' % i, t.ctypes_repr) )
    
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


_closure_type_cache = {}
def make_closure_type(untyped_fn, closure_arg_types = []):
  name = untyped_fn.name
  closure_arg_types = tuple(closure_arg_types)
  key = (name, closure_arg_types)
  if key in _closure_type_cache:
    return _closure_type_cache[key]
  else:
    t = ClosureT(name, closure_arg_types)
    _closure_type_cache[key] = t 
    return t
  
from types import FunctionType
import ast_conversion 

def typeof(f):
  untyped_fn = ast_conversion.translate_function_value(f)
  closure_args = untyped_fn.get_closure_args(untyped_fn)
  closure_arg_types = map(type_conv.typeof, closure_args)
  return make_closure_type(untyped_fn, closure_arg_types)
  
   
  
    


type_conv.register(FunctionType)
  
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



###########################################
#
#  Pointers! 
#
###########################################    
    
class PtrT(ConcreteT):
  """
  I'm not giving pointer a concrete to_python/from_python
  conversions or any usable fields, so it's up to our ctypes_repr
  and llvm_backend to appropriately interpret objects of this type. 
  """
  _members = ['elt_type']
  
  def index_type(self, idx):
    assert isinstance(idx, IntT), "Index into pointer must be of type int, got %s" % (idx)
    return self.elt_type
  
  def node_init(self):
    self._ctypes_repr = ctypes.POINTER(self.elt_type.ctypes_repr)
  
  def __str__(self):
    return "ptr(%s)" % self.elt_type
  
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

