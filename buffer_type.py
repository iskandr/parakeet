import ctypes 
from core_type import ptr 
from scalar_types import Int64 

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
    
    self._fields_ = [
      ('pointer', ptr(self.elt_type)), 
      ('length', Int64)                 
    ]
    self.ctypes_pointer_t = ctypes.POINTER(self.elt_type.ctypes_repr)
    class BufferRepr(ctypes.Structure):
      _fields_ = [
        ('pointer', self.ctypes_pointer_t),
        ('length', ctypes.c_int64)
      ]   
    self.ctypes_repr = BufferRepr  

  
  def to_ctypes(self, x):
    assert isinstance(x, buffer)
    ptr, length = buffer_info(x, self.ctypes_pointer_t)
    return self.ctypes_repr(ptr, length)
  
  def from_ctypes(self, x):
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

