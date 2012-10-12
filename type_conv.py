import ctypes 
import dtypes   


  
 
_typeof = {}
def typeof(python_value):
  python_type = type(python_value)
  assert python_type in _typeof, \
    "No type converter found for %s : %s" % (python_value, python_type) 
  return _typeof[python_type](python_value)



def from_python(python_value):
  """
  Look up the ctypes representation of the
  corresponding parakeet type and call the 
  converter with the ctypes class and python value
  """
  
  parakeet_type = typeof(python_value)
  result = parakeet_type.from_python(python_value)
  print "FROM_PYTHON", python_value, result 
  return result 


def to_python(internal_value, parakeet_type):
  result = parakeet_type.from_python(internal_value)
  print "TO_PYTHON", internal_value, parakeet_type, result 
  return result 


def register(python_type, typeof):
  """
  Map each python type to either a parakeet type or a function that returns a parakeet type
  """
  _typeof[python_type] = typeof
   
  
  