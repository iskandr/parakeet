

# SSA ID -> untyped function 
untyped_functions = {}

def lookup_untyped(untyped_name):
  if isinstance(untyped_name, str):
    return untyped_functions[untyped_name]
  else:
    assert hasattr(untyped_name, 'body'), \
      'Expected a function, got %s' % untyped_name
    return untyped_name 

# SSA ID -> typed function  
typed_functions = {}

def is_typed(fn_name):
  assert isinstance(fn_name, str), \
    "Expected function name, got: " + str(fn_name)
  return fn_name in typed_functions
  
# (untyped ID, arg types) -> typed FunDef
specializations = {} 
    
# python value of a user-defined function mapped to its untyped ID  
known_python_functions = {}

def register_python_fn(fn_val, fundef):
  known_python_functions[fn_val] = fundef.name

def already_registered_python_fn(fn_val):
  return fn_val in known_python_functions

def lookup_python_fn(fn_val):
  """Returns untyped function definition"""
  
  return untyped_functions[known_python_functions[fn_val] ]
  
def lookup_python_fn_dependencies(fn_val):
  info = known_python_functions[fn_val]
  return [info.globals_dict[n] for n in info.dependency_names]


def add_specialization(untyped_id, arg_types, typed_fundef):
  key = (untyped_id, tuple(arg_types))
  assert key not in specializations
  specializations[key] = typed_fundef 
  typed_functions[typed_fundef.name] = typed_fundef  

def find_specialization(untyped_id, arg_types):
  key = (untyped_id, tuple(arg_types))
  return specializations[key]

class ClosureSignatures:
  """
  Map each (untyped fn id, fixed arg) types to a distinct integer
  so that the runtime representation of closures just need to 
  carry this ID
  """
  closure_sig_to_id = {}
  id_to_closure_sig = {}
  max_id = 0  
  
  @classmethod
  def get_id(cls, closure_sig):
    if closure_sig in cls.closure_sig_to_id:
      return cls.closure_sig_to_id[closure_sig]
    else:
      num = cls.max_id
      cls.max_id += 1
      cls.closure_sig_to_id[closure_sig] = num
      return num 
  
  @classmethod     
  def get_closure_signature(cls, num):
    return cls.id_to_closure_sig[num]

