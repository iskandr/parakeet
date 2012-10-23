
import numpy as np 
import ctypes
 
from tuple_type import repeat_tuple
from core_types import StructT, IncompatibleTypes, ScalarT, Int64, ptr_type 



def buffer_info(buf, ptr_type = ctypes.c_void_p):    
  """Given a python buffer, return its address and length"""
  assert isinstance(buf, buffer)
  address = ptr_type()
  length = ctypes.c_ssize_t() 
  obj =  ctypes.py_object(buf) 
  ctypes.pythonapi.PyObject_AsReadBuffer(obj, ctypes.byref(address), ctypes.byref(length))
  return address, length   


class ArrayT(StructT):
  _members = ['elt_type', 'rank']

  def finalize_init(self):
    assert isinstance(self.elt_type, ScalarT)
    tuple_t = repeat_tuple(Int64, self.rank)
    self.shape_t = tuple_t
    self.strides_t = tuple_t
    self.ptr_t = ptr_type(self.elt_type) 
    self._field_types = [
      ('data', self.ptr_t), 
      ('shape', tuple_t), 
      ('strides', tuple_t),
    ]

  def dtype(self):
    return self.elt_type.dtype
 
  def __eq__(self, other): 
    return isinstance(other, ArrayT) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)


  def from_python(self, x):
    assert isinstance(x, np.ndarray)
    
    data = x.data 
    ctypes_elt_t = self.elt_t.ctypes_repr
    ctypes_ptr_t = ctypes.POINTER(ctypes_elt_t) 
    ptr, length = buffer_info(data, ctypes_ptr_t)
    return self.ctypes_repr(ptr, length)
    
  def to_python(self, obj):
    """
    For now, to avoid to dealing with the messiness of ownership,  
    we just always copy data on the way out of Parakeet
    """
    shape = self.shape_t.to_python(obj.shape)
    strides = self.strides_t.to_python(obj.strides)
    elt_size = self.elt_type.nbytes
    assert any([stride == elt_size for stride in strides]), "Discontiguous array not yet supported"
    n_elts = sum(shape)
    n_bytes = n_elts * elt_size 
    dest_buf = ctypes.pythonapi.PyBuffer_New(n_bytes)
    dest_ptr, _ = buffer_info(dest_buf, self.ptr_t.ctypes_repr)
    
    # copy data from pointer
    ctypes.memmove(dest_ptr, self.data, n_bytes)
    
    return np.ndarray(shape, dtype = self.elt_type.dtype, dest_buf, strides = strides) 
    
_array_types = {}
def array_type(elt_t, rank):
  key = (elt_t, rank) 
  if key in _array_types:
    return _array_types[key]
  else:
    _array_types[key] = ArrayT(elt_t, rank)