import ast 
import ssa 

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
  def __init__(self, current_scope = None,  outer_env = None):
    if current_scope is None:
      current_scope = {}
    
    if outer_env is None:
      outer_env = {}
      
    self.scopes = [current_scope]
    
    # a top-level function will point to an environment for globals
    # and a nested function points to the environment of its enclosing fn
    self.outer_env = outer_env
     
    # the set of accessed nonlocal variables is shared between all scopes
    self.nonlocals = set([])
    
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

  def in_outer_env(self, key):
    """Recursively move up the chain of 'outer_env' links to find if
    a variable has been bound outside the current function
    """
    if self.outer_env is None:
      return False
    elif key in self.outer_env:
      return True
    elif isinstance(self.outer_env, ScopedEnv):
      return self.outer_env.in_outer_env(key)
    else:
      return False 

#class EnvChain:
#  def __init__(self, current = None, parent = None):
#    if current is None:
#      current = ScopedEnv()
#    self.current = current 
#    self.parent = parent 
  
#  def __getitem__(self, key):
#    if key in self.current:
#      return self.current[key]
#    elif self.parent is not None:
#      return self.parent[key]
#    else:
 #     raise NameNotFound(key)
    
 # def __contains__(self, key):
 #   return key in self.current or \
 #     (self.parent is not None and key in self.parent)    

empty_set = set([])

def translate_Name(name, env):
  """
  Convert a variable name to its versioned SSA identifier and 
  if the name isn't local return it in a one-element set denoting
  which nonlocals get accessed
  """
  if name in env:
    return ssa.Var(env[name]), empty_set
  # is it at least somewhere in the chain of outer scopes?  
  elif env.in_outer_env(name):
    ssa_name = env.fresh(name) 
    return ssa.Var(ssa_name), set([name])
  else:
    raise NameNotFound(name)
    
def translate_BinOp(name, op, left, right, env):
  ssa_left, left_nonlocals = translate_expr(left, env)
  ssa_right, right_nonlocals = translate_expr(right, env)
  nonlocals = left_nonlocals.union(right_nonlocals)
  ssa.Binop(op, ssa_left, ssa_right), nonlocals
  

def translate_expr(expr, env):
  if isinstance(expr, ast.BinOp):
    return translate_BinOp(expr.op, expr.left, expr.right, env)
  elif isinstance(expr, ast.Name):
    return translate_Name(expr.id, env)
    
  elif isinstance(expr, ast.Num):
    ssa.Const(expr.n), empty_set 
    
def translate_Assign(lhs, rhs, env):
  assert isinstance(lhs, ast.Name)
  ssa_lhs_id = env.fresh(lhs.id) 
  ssa_rhs, nonlocals = translate_expr(rhs, env)
  return ssa.Assign(ssa_lhs_id, ssa_rhs), nonlocals
    
def translate_Return(value, env):
  ssa_value, nonlocals = translate_expr(value, env)
  return ssa.Return(ssa_value), nonlocals
def translate_stmt(stmt, env):
  """
  Given a stmt, dispatch based on its class type to a particular
  translate_ function and return the set of nonlocal vars accessed
  by this statment
  """
  if isinstance(stmt, ast.FunctionDef):
    return translate_FunctionDef(stmt.name, stmt.args, stmt.body, env)
  elif isinstance(stmt, ast.Assign):
    
    return translate_Assign(stmt.target[0], stmt.value, env)
  elif isinstance(stmt, ast.Return):
    return translate_Return(stmt.value)


def translate_module(m, env = None):
  assert isinstance(m, ast.Module)
  if env is None: 
    env = ScopedEnv()

  # module should contain only function definitions 
  for stmt in m.body:
    translate_stmt(stmt, env)


def translate_FunctionDef(name, args, body, global_env = None):
  """
  - translate the body of the function and collect every global variable
  which gets referenced, giving it a new SSA ID upon first reference
  - and then....
  """
  ssa_args = dict(zip(args, map(NameSupply.fresh, args)))
  env = ScopedEnv(ssa_args, global_env = global_env)
  
    