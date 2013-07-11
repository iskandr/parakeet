from syntax import Fn

# python value of a user-defined function mapped to its
# untyped representation
known_python_functions = {}

def register_python_fn(fn_val, fundef):
  known_python_functions[fn_val] = fundef

def already_registered_python_fn(fn_val):
  return fn_val in known_python_functions

def lookup_python_fn(fn_val):
  """Returns untyped function definition"""

  return known_python_functions[fn_val]
