import ast
import inspect  
import syntax 
import prims 
from prims import is_prim, find_prim_from_python_value
from function_registry import untyped_functions, already_registered_python_fn
from function_registry import register_python_fn, lookup_python_fn, lookup_prim_fn 
import names
from names import NameNotFound 
from scoped_env import ScopedEnv 
from common import dispatch

def extract_arg_names(args):
  assert not args.vararg
  assert not args.kwarg
  assert not args.defaults
  return [arg.id for arg in args.args]

def collect_defs_from_node(node):
  """
  Recursively traverse nested statements and collect 
  variable names from the left-hand-sides of assignments,
  ignoring variables if they appear within a slice or index
  """
  if isinstance(node, ast.While):
    return collect_defs_from_list(node.body + node.orelse)
  elif isinstance(node, ast.If):
    return collect_defs_from_list(node.body + node.orelse)
  elif isinstance(node, ast.Assign):
    return collect_defs_from_list(node.targets)
  elif isinstance(node, ast.Name):
    return set([node.id])
  elif isinstance(node, ast.Tuple):
    return collect_defs_from_list(node.elts)
  else:
    return set([])
        
def collect_defs_from_list(nodes):
  assert isinstance(nodes, list)
  defs = set([])
  for node in nodes:
    defs.update(collect_defs_from_node(node))
  return defs 


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
          prim = find_prim_from_python_value(global_value)
          prim_fundef = lookup_prim_fn(prim)
          return syntax.Closure(prim_fundef.name, [])
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
    elif name == 'True':
      return syntax.Const(True)
    elif name == 'False':
      return syntax.Const(False)
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
  
  def create_phi_nodes(left_scope, right_scope, new_names = {}):
    """
    Phi nodes make explicit the possible sources of each variable's values and 
    are needed when either two branches merge or when one was optionally taken. 
    """
    merge = {}
    for (name, ssa_name) in left_scope.iteritems():      
      left = syntax.Var(ssa_name)
      if name in right_scope:
        right = syntax.Var (right_scope[name])
      else:
        right = translate_Name(name)
        
      if name in new_names:
        new_name = new_names[name]
      else:
        new_name = env.fresh(name)
      merge[new_name] = (left, right)
      
    for (name, ssa_name) in right_scope.iteritems():
      if name not in left_scope:
        left = translate_Name(name)
        right = syntax.Var(ssa_name)
    
        if name in new_names:
          new_name = new_names[name]
        else:
          new_name = env.fresh(name)
        merge[new_name] = (left, right)
    return merge 
   
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
    
    def translate_Num():
      return syntax.Const(expr.n)
    
    def translate_IfExp():
      temp1, temp2, result = \
         map(env.fresh_var, ["if_true", "if_false", "if_result"])
      cond = translate_expr(expr.test)
      true_block = [syntax.Assign(temp1, translate_expr(expr.body))]
      false_block = [syntax.Assign(temp2, translate_expr(expr.orelse))]
      merge = {result.name :  (temp1, temp2)}
      if_stmt = syntax.If(cond, true_block, false_block, merge) 
      env.current_block().append(if_stmt)
      return result
    
    if isinstance(expr, ast.Name):
      # name is a special case since its translation function needs to be accessed
      # from outside translate_expr 
      return translate_Name(expr.id)
    else:
      result = dispatch(expr, 'translate')
      assert isinstance(result, syntax.Expr), "%s not an expr" % result 
      return result 
      


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
      lhs = stmt.targets[0]
      assert isinstance(lhs, ast.Name)
      # important to evaluate RHS before LHS for statements like 'x = x + 1' 
      ssa_rhs = translate_expr(stmt.value)
      ssa_lhs = env.fresh_var(lhs.id) 
      return syntax.Assign(ssa_lhs, ssa_rhs)
    elif isinstance(stmt, ast.Return):
      rhs = syntax.Return(translate_expr(stmt.value))
      return rhs 
    elif isinstance(stmt, ast.If):
      cond = translate_expr(stmt.test)
      true_scope, true_block  = translate_block(stmt.body)
      #print true_block  
      false_scope, false_block = translate_block(stmt.orelse)
      merge = create_phi_nodes(true_scope, false_scope)
      return syntax.If(cond, true_block, false_block, merge)
   
    elif isinstance(stmt, ast.While):
      assert stmt.orelse == [], "Expected empty orelse block, got: %s" % stmt.orelse 


      # push a scope for the version of variables appearing within the loop 
      # create a new version for each var defined in the loop 
      env.push()
      for lhs_var in collect_defs_from_list(stmt.body):
        env.fresh(lhs_var)
      # evaluate the condition in the context of the version of loop variables we see 
      # at the start of the loop 
      cond = translate_expr(stmt.test)
      # translate_block pushes an additional env which will track 
      # the versions of variables throughout the loop, so we get back
      # a dict with the last version of each variable 
      loop_end_scope, body = translate_block(stmt.body)
      loop_start_scope, _ = env.pop()
      # given empty scope for right branch so we always merge with version of variable 
      # before loop started 
      #assert "counter" not in env, [translate_Name("counter"), body]
      merge_before = create_phi_nodes( {}, loop_end_scope, new_names = loop_start_scope)
     
      # don't provide a new_names dict so that fresh versions after the loop are created 
      # for each var in the current env
      merge_after = create_phi_nodes({}, loop_end_scope)

      return syntax.While(cond, body, merge_before, merge_after)
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
  

  
  
  