import names
import syntax 
# SSA ID -> untyped FunDef
untyped_functions = {}

# (untyped ID, arg types) -> typed FunDef
typed_functions = {}


# every prim is associated with an untyped function 
# whose body consists only of calling that prim 
untyped_prim_functions = {}
    
def lookup_prim_fn(p):
  """Given a primitive, return an untyped function which calls that prim"""
  if p in untyped_prim_functions:
    return untyped_prim_functions
  else:
    fn_name = names.fresh(p.name)
    args = [syntax.Var(x) for x in names.fresh_list(p.nin)]
    result = names.fresh("result")
    body = [syntax.Assign(syntax.Var(result), syntax.PrimCall(p, args))]
    fundef = syntax.Fn(fn_name, args, body, [])
    untyped_prim_functions[p] = fundef
    untyped_functions[fn_name] = fundef
    return fundef 

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

