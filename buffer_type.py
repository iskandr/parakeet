


  
import ctypes 
from core_types import PtrT, Int64, StructT

  
def buffer_info(buf, ptr_type = ctypes.c_void_p):    
  """Given a python buffer, return its address and length"""
  assert isinstance(buf, buffer)
  address = ptr_type()
  length = ctypes.c_ssize_t() 
  obj =  ctypes.py_object(buf) 
  ctypes.pythonapi.PyObject_AsReadBuffer(obj, ctypes.byref(address), ctypes.byref(length))
  return address, length   


class BufferT(StructT):
  """
  Wrap a python buffer, which combines a pointer and its data size
  """ 
  _members = ['elt_type']
  
  
  def node_init(self):
    self._fields_ = [
      ('pointer', PtrT(self.elt_type)), 
      ('length', Int64)                 
    ]
    
  def from_python(self, x):
    assert isinstance(x, buffer)
    
    ctypes_elt_t = self.elt_t.ctypes_repr
    ctypes_ptr_t = ctypes.POINTER(ctypes_elt_t) 
    ptr, length = buffer_info(x, ctypes_ptr_t)
    return self.ctypes_repr(ptr, length)
    
  def to_python(self, internal_buffer_obj):
    """
    For now, to avoid to dealing with the messiness of ownership,  
    we just always copy data on the way out of Parakeet
    """
    
    dest_buf = ctypes.pythonapi.PyBuffer_New(internal_buffer_obj.length)
    ctypes_elt_t = self.elt_t.ctypes_repr
    ctypes_ptr_t = ctypes.POINTER(ctypes_elt_t)
    dest_ptr, _ = buffer_info(dest_buf, ctypes_ptr_t)
    
    # copy data from pointer
    ctypes.memmove(dest_ptr, internal_buffer_obj.pointer, internal_buffer_obj.length)
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
