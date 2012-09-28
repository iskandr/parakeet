import ast
import inspect  
import syntax 
import prims 
from prims import is_prim, find_prim
from function_registry import untyped_functions, already_registered_python_fn
from function_registry import register_python_fn, lookup_python_fn 
import names
from names import NameNotFound

  

class ScopedEnv:  
  def __init__(self, outer_env = None):
    self.scopes = [{}]
    self.blocks = [[]]
    # link together environments of nested functions
    self.outer_env = outer_env
    
  def fresh(self, name):
    fresh_name = names.fresh(name)
    self.scopes[-1][name] = fresh_name 
    return fresh_name
  
  def fresh_var(self, name):
    return syntax.Var(self.fresh(name))
  
  def push(self, scope = None, block = None):
    if scope is None:
      scope = {}
    if block is None:
      block = []
    self.scopes.append(scope)
    self.blocks.append(block)
  
  def pop(self):
    scope = self.scopes.pop()
    block = self.blocks.pop()
    return scope, block 
  
  def current_scope(self):
    return self.scopes[-1]
  
  def current_block(self):
    return self.blocks[-1]
  
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

def extract_arg_names(args):
  assert not args.vararg
  assert not args.kwarg
  assert not args.defaults
  return [arg.id for arg in args.args]



def translate_FunctionDef(name,  args, body, global_values, outer_value_env = None):
   
  # external names of the nonlocals we use
  nonlocal_original_names =  []
  # syntax names for nonlocals which should get passed in
   
  nonlocal_arg_names = []
  arg_names = extract_arg_names(args)
  env = ScopedEnv()
  ssa_arg_names = [env.fresh(n) for n in arg_names]
  
  # maps a local SSA ID to a global fn id paired with some closure values

  def global_ref(name):
    """
    A global is:
      (1) data which needs to be added to the list of nonlocals we depend on 
      (2) a primitive fn
      (3) a user-defined fn which should be translated
    """ 
    if name in global_values:
      global_value = global_values[name]
   
      if hasattr(global_value, '__call__'):
        if is_prim(global_value): 
          return syntax.Prim(find_prim(global_value))
        elif already_registered_python_fn(global_value):
          fundef = lookup_python_fn(global_value)
          ssa_name = fundef.name 
          closure_args = map(translate_Name, fundef.nonlocals)
          return syntax.Closure(ssa_name, closure_args)
        else:
          # we expect that translate_function_value will add 
          # the function to the global lookup table known_functions
          ssa_fundef = translate_function_value(global_value)
          return syntax.Closure (ssa_fundef.name, [])
        
      else:
        
        # if it's global data... 
        ssa_name = env.fresh(name)
        nonlocal_original_names.append(name)
        nonlocal_arg_names.append(ssa_name)
        #print name, ssa_name, nonlocal_arg_names 
        return syntax.Var (ssa_name)
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
      return syntax.Var(env[name])
    # is it at least somewhere in the chain of outer scopes?  
    else:
      outer = env.recursive_lookup(name, skip_current = True)
      if outer:
        nonlocal_original_names.add(name)  
        ssa_name = env.fresh(name) 
        nonlocal_arg_names.append(ssa_name)
        return syntax.Var(ssa_name)
      else:
        return global_ref(name)
      
  def translate_expr(expr):
    def translate_UnaryOp():
      ssa_val = translate_expr(expr.operand)
      prim = prims.find_ast_op(expr.op)
      return syntax.PrimCall(prim, [ssa_val])
    
    def translate_BinOp():
      ssa_left = translate_expr(expr.left)
      ssa_right = translate_expr(expr.right)
      prim = prims.find_ast_op(expr.op)
      return syntax.PrimCall(prim, [ssa_left, ssa_right] )
     
    def translate_Compare():
      lhs = translate_expr(expr.left)   
      assert len(expr.ops) == 1
      prim = prims.find_ast_op(expr.ops[0])
      assert len(expr.comparators) == 1
      rhs = translate_expr(expr.comparators[0])
      return syntax.PrimCall(prim, [lhs, rhs])
    
    def translate_Call():
      fn, args, kwargs, starargs = \
        expr.func, expr.args, expr.kwargs, expr.starargs
      assert kwargs is None, "Dictionary of keyword args not supported"
      assert starargs is None, "List of varargs not supported"
      fn_val = translate_expr(fn)
      arg_vals = map(translate_expr, args)
      return syntax.Invoke(fn_val, arg_vals) 
    
    def translate_IfExp():
      temp1, temp2, result = \
         map(env.fresh_var, ["if_true", "if_false", "if_result"])
      cond = translate_expr(expr.test)
      true_block = [syntax.Assign(temp1, translate_expr(expr.body))]
      false_block = [syntax.Assign(temp2, translate_expr(expr.orelse))]
      merge = {result :  (temp1, temp2)}
      if_stmt = syntax.If(cond, true_block, false_block, merge) 
      env.current_block().append(if_stmt)
      return syntax.Var(result)
      
    nodetype = expr.__class__.__name__
    if isinstance(expr, ast.Name):
      return translate_Name(expr.id)
    elif isinstance(expr, ast.Num):
      return syntax.Const(expr.n)
    else:
      translator_fn_name = 'translate_' + nodetype
      translate_fn = locals()[translator_fn_name]
      result = translate_fn()
      assert isinstance(result, syntax.Expr), "%s not an expr" % result 
      return result 
      
  def translate_Assign(lhs, rhs):
    assert isinstance(lhs, ast.Name)
    
    ssa_lhs = env.fresh_var(lhs.id) 
    ssa_rhs = translate_expr(rhs)
    return syntax.Assign(ssa_lhs, ssa_rhs)
      

  def translate_stmt(stmt):
    """
    Given a stmt, dispatch based on its class type to a particular
    translate_ function and return the set of nonlocal vars accessed
    by this statment
    """
    if isinstance(stmt, ast.FunctionDef):
      name, args, body = stmt.name, stmt.args, stmt.body
      fundef = \
        translate_FunctionDef(name, args, body, global_values, env)
      closure_args = map(translate_Name, fundef.nonlocals)
      local_name = env.fresh_var(name)
      closure = syntax.Closure(fundef.name, closure_args)
      return syntax.Assign(local_name, closure)
    
    elif isinstance(stmt, ast.Assign):     
      return translate_Assign(stmt.targets[0], stmt.value)
    elif isinstance(stmt, ast.Return):
      rhs = syntax.Return(translate_expr(stmt.value))
      return rhs 
    elif isinstance(stmt, ast.If):
      cond = translate_expr(stmt.test)
      true_scope, true_block  = translate_block(stmt.body)
      #print true_block  
      false_scope, false_block = translate_block(stmt.orelse)
      merge = {}
      
      for (name, ssa_name) in true_scope.iteritems():
        new_name = env.fresh(name)
        left = syntax.Var(ssa_name)
        if name in false_scope:
          right = syntax.Var (false_scope[name])
        else:
          right = translate_Name(name)
        merge[new_name] = (left,right)
      for (name, ssa_name) in false_scope.iteritems():
        if name not in true_scope:
          new_name = env.fresh(name)
          left = translate_Name(name)
          right = syntax.Var(ssa_name)
          merge[new_name] = (left, right)
      return syntax.If(cond, true_block, false_block, merge)
   
    elif isinstance(stmt, ast.While):
      raise RuntimeError("While loops not implemented")
    elif isinstance(stmt, ast.For):
      return RuntimeError("For loops not implemneted")
    else:
      raise RuntimeError("Not implemented: %s"  % stmt)
  

  
  def translate_block(stmts):
    env.push()
    curr_block = env.current_block()
    for stmt in stmts:
      curr_block.append(translate_stmt(stmt))
    return env.pop()
    
  _, ssa_body = translate_block(body)   
  ssa_fn_name = names.fresh(name)
  full_args = nonlocal_arg_names + ssa_arg_names
  #print ssa_fn_name, full_args 
  fundef = syntax.Fn(ssa_fn_name, full_args, ssa_body, nonlocal_original_names)
  untyped_functions[ssa_fn_name]  = fundef 
  return fundef

def translate_module(m, global_values, outer_env = None):
  assert isinstance(m, ast.Module)
  assert len(m.body) == 1
  assert isinstance(m.body[0], ast.FunctionDef)
  fundef = m.body[0]
  name, args, body = fundef.name, fundef.args, fundef.body  
  return translate_FunctionDef(name, args, body, global_values, outer_env)

def translate_function_source(source, global_values):
  syntax = ast.parse(source)
  return translate_module(syntax, global_values)

def translate_function_value(fn):
  if already_registered_python_fn(fn):
    return lookup_python_fn(fn)
  else:
    assert hasattr(fn, 'func_globals')
    source = inspect.getsource(fn)
    fundef = translate_function_source(source, fn.func_globals)
    register_python_fn(fn, fundef)
    return fundef   
  

  
  
  