_type_mapping = {}
_typeof_functions = {}

def register(python_types, parakeet_type, typeof = None):
  """
  Map each python type to either a parakeet type or a function that returns a
  parakeet type
  """

  if typeof is None:
    typeof = lambda _: parakeet_type

  if not isinstance(python_types, (list, tuple)):
    python_types = [python_types]

  for python_type in python_types:
    _typeof_functions[python_type] = typeof
    _type_mapping[python_type] = parakeet_type

def equiv_type(python_type):
  assert python_type in _type_mapping, \
      "No type mapping found for %s" % python_type
  return _type_mapping[python_type]

def typeof(python_value):
  python_type = type(python_value)
  assert python_type in _typeof_functions, \
      "Don't know how to convert value %s : %s" % (python_value, python_type)
  return _typeof_functions[python_type](python_value)

def from_python(python_value):
  """
  Look up the ctypes representation of the corresponding parakeet type and call
  the converter with the ctypes class and python value
  """

  parakeet_type = typeof(python_value)
  result = parakeet_type.from_python(python_value)
  return result

def to_python(internal_value, parakeet_type):
  result = parakeet_type.from_python(internal_value)
  return result
