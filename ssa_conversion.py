import ast
import inspect  
import ssa 
from prims import prims 

class NameNotFound(Exception):
  def __init__(self, name):
    self.name = name 
    
class NameSupply:
  versions = {}
  original_names = {}
  
  @staticmethod
  def get(self, name):
    version = self.versions.get(name)
    if version is None:
      raise NameNotFound(name)
    else:
      return "%s.%d" % (name, version)
    
  @staticmethod  
  def fresh(self, name):
    version = self.versions.get(name, 0) + 1 
    self.versions[name] = version
    ssa_name = "%s.%d" % (name, version)
    self.original_names[ssa_name] = name
    return ssa_name 
  


class ScopedEnv:  
  def __init__(self, current_scope = None, outer_env = None):
    if current_scope is None:
      current_scope = {}

    self.scopes = [current_scope]
    # link together environments of nested functions
    self.outer_env = outer_env
    
  def fresh(self, name):
    fresh_name = NameSupply.fresh(name)
    self.scopes[-1][name] = fresh_name 
    return fresh_name
  
  def push_scope(self, scope = None):
    if scope is None:
      scope = {}
    self.scopes.append(scope)
  
  def pop_scope(self):
    return self.scopes.pop()
  
  def __getitem__(self, key):
    for scope in reversed(self.scopes):
      if key in scope:
        return scope[key]
    raise NameNotFound(key)

  def __contains__(self, key):
    for scope in reversed(self.scopes):
      if key in scope: 
        return True
    return False 
  
  def recursive_lookup(self, key, skip_current = False):
    if not skip_current and key in self:
      return self[key]
    else:
      if self.outer_env:
        self.outer_env.recursive_lookup(key)
      else:
        return None

# maps python value of a user-defined function 
# to a FunDef 
known_functions = {}

def translate_FunctionDef(name, body, args, global_values, outer_env = None):
   
  nonlocals =  set([])
  ssa_args = dict(zip(args, map(NameSupply.fresh, args)))
  env = ScopedEnv(current_scope = ssa_args, outer_env = outer_env)
  
  
  def global_fn_ref(global_value):
    if global_value in prims: 
      return ssa.Prim(global_value.__name__)
    elif global_value in known_functions:
      ssa_name = known_functions[global_value].ssa_name 
      return ssa.FnRef (ssa_name)
    else:
      ssa_fundef = translate_function_value(global_value)
      known_functions[global_value] = ssa_fundef  
      return ssa.FnRef (ssa_fundef.ssa_name)
    
  def global_ref(name):
    if name in global_values:
      global_value = global_values[name]
      if hasattr(global_value, '__call__'):
        return global_fn_ref(global_value)
      else:
        raise RuntimeError("global value not a function")   
    else:
      raise NameNotFound(name)
  def translate_Name(name):
    """
    Convert a variable name to its versioned SSA identifier and 
    if the name isn't local it must be one of:
      (a) global data which needs to be added as an argument to this fn
      (b) a user-defined function which needs to be registered with parakeet
      (c) a primitive fn 
    """
    if name in env:
      return ssa.Var(env[name])
    # is it at least somewhere in the chain of outer scopes?  
    else:
      outer = env.recursive_lookup(name, skip_current = True)
      if outer:
        nonlocals.add(name)  
        ssa_name = env.fresh(name) 
        return ssa.Var(ssa_name)
      else:
        return global_ref(name)
      
  def translate_BinOp(name, op, left, right, env):
    ssa_left = translate_expr(left)
    ssa_right = translate_expr(right)
    ssa.Binop(op, ssa_left, ssa_right)
    
  
  def translate_expr(expr, env):
    if isinstance(expr, ast.BinOp):
      return translate_BinOp(expr.op, expr.left, expr.right)
    elif isinstance(expr, ast.Name):
      return translate_Name(expr.id)
      
    elif isinstance(expr, ast.Num):
      ssa.Const(expr.n) 
      
  def translate_Assign(lhs, rhs, env):
    assert isinstance(lhs, ast.Name)
    ssa_lhs_id = env.fresh(lhs.id) 
    ssa_rhs = translate_expr(rhs, env)
    return ssa.Assign(ssa_lhs_id, ssa_rhs)
      

  def translate_stmt(stmt, env):
    """
    Given a stmt, dispatch based on its class type to a particular
    translate_ function and return the set of nonlocal vars accessed
    by this statment
    """
    if isinstance(stmt, ast.FunctionDef):
      name, args, body = stmt.name, stmt.args, stmt.body
      fundef = translate_FunctionDef(name, args, body, global_values, env)
      nonlocals.update(fundef.nonlocals)
      # update some global table of defined functions? 
      # give the function a unique SSA ID? 
    elif isinstance(stmt, ast.Assign):     
      return translate_Assign(stmt.target[0], stmt.value)
    elif isinstance(stmt, ast.Return):
      return ssa.Return(translate_expr(stmt.value))
  
  
  ssa_body = [translate_stmt(stmt) for stmt in body]
  # should I register the function globally now? 
  return {'body': body, 'nonlocals':nonlocals}

def translate_module(m, global_values, outer_env = None):
  assert isinstance(m, ast.Module)
  assert len(m.body) == 1
  assert isinstance(m.body[0], ast.FunctionDef)
  fundef = m.body[0]
  name, args, body = fundef.name, fundef.args, fundef.body 
  return translate_FunctionDef(name, args, body, global_values, outer_env)

def translate_function_source(source):
  syntax = ast.parse(source)
  return translate_module(syntax)

def translate_function_value(fn):
  source = inspect.getsource(fn)
  return translate_function_source(source)
  

  
  
  