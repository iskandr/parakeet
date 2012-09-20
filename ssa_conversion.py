import ssa 

class ScopedEnv:
  def __init__(self, parent = None):
    self.current_scope = {}
    self.parent = parent 
  
  def __getitem__(self, key):
    if key in self.current_scope:
      return self.current_scope[key]
    elif self.parent is not None:
      return self.parent[key]
    else:
      raise RuntimeError("%s not found" % key)
      
import ast 

def translate_fundef(name, args, body, visible_vars):
  """
  - translate the body of the function and collect every global variable
  which gets referenced, giving it a new SSA ID upon first reference
  - and then....
  """
  
def translate_expr(expr, env):
  if isinstance(expr, ast.Binop):
    left, left_nonlocals = translate_expr(expr.left, env)
    right, right_nonlocals = translate_expr(expr.right, env)
    nonlocals = left_nonlocals.union(right_nonlocals)
    Binop(expr.op, left, right, left), nonlocals
    
  elif isinstance(expr, ast.Name):
    if ast.id in env.current_scope:
    expr.id
  elif isinstance(expr, ast.Num):
    expr.n 
    
def translate_stmt(stmt, env):
  """
  Given a stmt, dispatch based on its class type to a particular
  translate_ function and return the set of nonlocal vars accessed
  by this statment
  """
  if isinstance(stmt, ast.FunctionDef):
    return translate_function_def(stmt.name, stmt.args, stmt.body, env)
  elif isinstance(stmt, ast.Assign):
    return translate_assign(stmt.target[0], stmt.value, env)
  elif isinstance(stmt, ast.Return):
    return translate_return(stmt.value)


def translate_module(m, env = None):
  assert isinstance(m, ast.Module)
  if env is None: 
    env = ScopedEnv()

  # module should contain only function definitions 
  for stmt in m.body:
    translate_stmt(stmt, env)

  