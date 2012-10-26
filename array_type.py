
import numpy as np 
import ctypes
 
import tuple_type 
from core_types import StructT, IncompatibleTypes, ScalarT, Int64, ptr_type
import core_types 



def buffer_info(buf, ptr_type = ctypes.c_void_p):    
  """Given a python buffer, return its address and length"""
  assert isinstance(buf, buffer)
  address = ptr_type()
  length = ctypes.c_ssize_t() 
  obj =  ctypes.py_object(buf) 
  ctypes.pythonapi.PyObject_AsReadBuffer(obj, ctypes.byref(address), ctypes.byref(length))
  return address, length.value  

ctypes.pythonapi.PyBuffer_New.argtypes = (ctypes.c_ulong,)
ctypes.pythonapi.PyBuffer_New.restype = ctypes.py_object
AllocateBuffer = ctypes.pythonapi.PyBuffer_New

class ArrayT(StructT):
  _members = ['elt_type', 'rank']

  def node_init(self):
    assert isinstance(self.elt_type, ScalarT)
    tuple_t = tuple_type.repeat_tuple(Int64, self.rank)
    
    self.shape_t = tuple_t
    self.strides_t = tuple_t
    self.ptr_t = ptr_type(self.elt_type) 
    self._fields_ = [
      ('data', self.ptr_t), 
      ('shape', tuple_t), 
      ('strides', tuple_t),
    ]

  def dtype(self):
    return self.elt_type.dtype
 
  def __str__(self):
    return "array%d(%s)" % (self.rank, self.elt_type)
  
  def __eq__(self, other): 
    return isinstance(other, ArrayT) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    elif isinstance(other, core_types.ScalarT):
      elt_t = self.elt_type 
      combined_elt_t = elt_t.combine(other)
      return array_type(combined_elt_t, self.rank)
    elif isinstance(other, ArrayT):
      assert self.rank == other.rank
      combined_elt_t = self.elt_type.combine(other.elt_type)
      return array_type(combined_elt_t, self.rank)
    else:
      raise IncompatibleTypes(self, other)

  def index_type(self, idx):
    """
    Given the type of my indices, what type of array will result?
    """
    if not isinstance(idx, core_types.Type):
      idx = idx.type 
      
    if isinstance(idx, core_types.IntT):
      return array_type(self.elt_type, self.rank - 1) 
    elif isinstance(idx, core_types.NoneT):
      return array_type(self.elt_type, 1)
    elif isinstance(idx, ArrayT):
      assert idx.rank == 1
      # slicing out a subset of my rows doesn't change my type 
      return self 
    elif isinstance(idx, tuple_type.TupleT):
      raise RuntimeError("Indexing by tuples not yet supported")
    else:
      raise RuntimeError("Unsupported index type: %s" % idx)

  def from_python(self, x):
    x = np.asarray(x)
         
    ptr, buffer_length = buffer_info(x.data, self.ptr_t.ctypes_repr)
    nelts = sum(x.shape)
    elt_size = x.dtype.itemsize
    total_bytes = nelts * elt_size 
    assert total_bytes == buffer_length, \
      "Shape %s has %d elements of size %d (total = %d) but buffer has length %d bytes"  % \
        (x.shape, nelts, elt_size, total_bytes, buffer_length)
    ctypes_shape = self.shape_t.from_python(x.shape)
    
    ctypes_strides = self.strides_t.from_python(x.strides)

    return self.ctypes_repr(ptr, ctypes.pointer(ctypes_shape), ctypes.pointer(ctypes_strides))
    
  def to_python(self, obj):
    """
    For now, to avoid to dealing with the messiness of ownership,  
    we just always copy data on the way out of Parakeet
    """
    shape = self.shape_t.to_python(obj.shape.contents)
    strides = self.strides_t.to_python(obj.strides.contents)
    elt_size = self.elt_type.nbytes
    assert any([stride == elt_size for stride in strides]), "Discontiguous array not yet supported"
    n_elts = sum(shape)
    n_bytes = n_elts * elt_size 
    dest_buf = AllocateBuffer(n_bytes)
    dest_ptr, _ = buffer_info(dest_buf, self.ptr_t.ctypes_repr)
    
    # copy data from pointer
    ctypes.memmove(dest_ptr, obj.data, n_bytes)
    
    return np.ndarray(shape, dtype = self.elt_type.dtype, buffer = dest_buf, strides = strides)
   
_array_types = {}
def array_type(elt_t, rank):
  key = (elt_t, rank) 
  if key in _array_types:
    return _array_types[key]
  else:
    if rank == 0:
      t = elt_t
    else:
      t = ArrayT(elt_t, rank)
    _array_types[key] = t
    return t 

import type_conv
def typeof(x):
  elt_t = core_types.from_dtype(x.dtype)
  rank = len(x.shape)

  return array_type(elt_t, rank)
 
type_conv.register( (np.ndarray, list),  ArrayT, typeof)

