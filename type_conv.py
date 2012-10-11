import ctypes 
import dtypes   

_ctypes_cache = {}

def ctypes_repr(parakeet_type):
  if parakeet_type in _ctypes_cache:
    return _ctypes_cache[parakeet_type]
  
  
  if hasattr(parakeet_type, '_fields_'):
    print "PT", parakeet_type
    ctypes_fields = []
    
    for (field_name, parakeet_field_type) in parakeet_type._fields_:
      pair = field_name, ctypes_repr(parakeet_field_type)
      ctypes_fields.append(pair)

    class Repr(ctypes.Structure):
      _fields_ = ctypes_fields
    Repr.__name__ = parakeet_type.node_type() +"_Repr"
    result = Repr 
    
  elif hasattr(parakeet_type, 'dtype'):
    result = dtypes.to_ctypes(parakeet_type.dtype)
    
  else:
    assert hasattr(parakeet_type, 'elt_type')
    ctypes_elt_t = ctypes_repr(parakeet_type.elt_type)
    result = ctypes.POINTER(ctypes_elt_t)
    
  _ctypes_cache[parakeet_type] = result
  return result 
  
 
_typeof = {}
def typeof(python_value):
  python_type = type(python_value)
  assert python_type in _typeof, \
    "No type converter found for %s : %s" % (python_value, python_type) 
  return _typeof[python_type](python_value)

_from_python= {}
def from_python(python_value):
  """
  Look up the ctypes representation of the
  corresponding parakeet type and call the 
  converter with the ctypes class and python value
  """
  parakeet_type = typeof(python_value)
  ctypes_class = ctypes_repr(parakeet_type)
  python_type = type(python_value)
  converter = _from_python[python_type]
  return converter(ctypes_class, python_value)
  

_to_python = {}
def to_python(internal_value, parakeet_type):
  parakeet_type_class = type(parakeet_type)
  assert parakeet_type_class in _to_python, \
    "Don't know how to converet type %s back to python" % parakeet_type_class.__name__ 
  return _to_python[parakeet_type_class](internal_value, parakeet_type)


def register(python_type, parakeet_type_class, typeof = None, from_python = None, to_python = None):
  """
  Map each python type to:
    - typeof: either a parakeet type or a function that returns a parakeet type 
              for a given value
    - from_python: function that creates internal representation for a python value
    - to_python: function that creates python value from given the
                 internal representation and parakeet_type
  """ 
  
  if typeof is None:
    # assume that parakeet_type_class is a singleton type which doesn't 
    # depend on input data
    parakeet_type = parakeet_type_class()
    def default_typeof(_):
      return parakeet_type   
    typeof = default_typeof
  
  if from_python is None:
    def default_from_python(ctypes_class, python_value):
      if hasattr(python_value, '__iter__'):
        return ctypes_class(*[from_python(elt) for elt in python_value])
      else:
        return ctypes_class(python_value)
    from_python = default_from_python
  
  if to_python is None:
    def default_to_python(x, _):
      assert hasattr(x, 'value'), "Default to_python converter failed"
      return x.value 
    to_python = default_to_python

  _typeof[python_type] = typeof
  _from_python[python_type] = from_python
  _to_python[parakeet_type_class] = to_python  
   
  
  