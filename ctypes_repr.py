from core_types import  dtypes, ScalarT, PtrT 

import ctypes 




_ctypes_cache = {}

def to_ctypes(parakeet_type):
  if parakeet_type in _ctypes_cache:
    return _ctypes_cache[parakeet_type]
  
  if hasattr(parakeet_type, '_fields_'):
    ctypes_fields = []
    
    for (field_name, parakeet_type) in parakeet_type._fields_:
      pair = field_name, to_ctypes(parakeet_type)
      ctypes_fields.append(pair)

    class Repr(ctypes.Structure):
      _fields_ = ctypes_fields
    Repr.__name__ = parakeet_type.__name__
    result = Repr 
   
  elif isinstance(parakeet_type, PtrT):
    result = ctypes.POINTER(to_ctypes(parakeet_type.elt_type))
  else:
    assert isinstance(parakeet_type, ScalarT)
    result = dtypes_to_c_scalar_types[parakeet_type.dtype]
  
  _ctypes_cache[parakeet_type] = result
  return result 
  