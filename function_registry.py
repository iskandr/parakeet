
# SSA ID -> untyped FunDef
untyped_functions = {}

# SSA ID -> typed FunDef
typed_functions = {}



class PythonFnInfo:
  """Information necessary to actually run a python function which has 
  been wrapped by Parakeet:
   - the ID of its internal untyped representation
   - the globals dictionary from the definition site of the fn
   - the names of globals which must be passed in as args
  """ 
  def __init__(self, untyped_id, globals_dict, dependency_names):
    self.untyped_id = untyped_id
    self.globals_dict = globals_dict
    self.dependency_names = dependency_names 

# python value of a user-defined function mapped to its untyped ID  
known_python_functions = {}

def register_python_fn(fn_val, fundef):
  info = PythonFnInfo(fundef.name, fn_val.func_globals, fundef.nonlocals)
  known_python_functions[fn_val] = info

def already_registered_python_fn(fn_val):
  return fn_val in known_python_functions

def lookup_python_fn(fn_val):
  """Returns untyped function definition"""
  
  info = known_python_functions[fn_val] 
  return untyped_functions[info.untyped_id]
  
def lookup_python_fn_dependencies(fn_val):
  info = known_python_functions[fn_val]
  return [info.globals_dict[n] for n in info.dependency_names]
   


