import syntax
import ast_conversion 
from function_registry import untyped_functions

class ReturnValue(Exception):
  def __init__(self, value):
    self.value = value 


class Closure:
  def __init__(self, fn, fixed_args):
    self.fn = fn 
    self.fixed_args = fixed_args

  
  
def eval_fn(fn, *args):
  n_expected = len(fn.args)
  n_given = len(args)
  assert n_expected == n_given , \
    "Expected %d args, given %d: %s" % (n_expected, n_given, args)
  env = dict(zip(fn.args, args)) 
  def eval_expr(expr):    
    def expr_Const():
      return expr.value
    
    def expr_PrimCall():
      fn = expr.prim.fn 
      arg_vals = map(eval_expr, expr.args)
      return fn(*arg_vals)
    
    def expr_Call():
      fundef = untyped_functions[expr.fn]
      arg_vals = map(eval_expr, expr.args)
      return eval_fn(fundef, *arg_vals) 
    
    def expr_Prim():
      return expr.value.fn
    
    def expr_Var():
      return env[expr.name]
    def expr_Invoke():
      # for the interpreter Invoke and Call are identical since
      # we're dealing with runtime reprs for functions, prims, and 
      # closures which are just python Callables
      clos = eval_expr(expr.closure)
      arg_vals = map(eval_expr, expr.args)
      combined_arg_vals = clos.fixed_args + arg_vals
      return eval_fn(clos.fn, *combined_arg_vals)
      
    def expr_Closure():
      fundef = untyped_functions[expr.fn]
      closure_arg_vals = map(eval_expr, expr.args) 
      return Closure(fundef, closure_arg_vals)
    
    functions = locals()
    nodetype =  expr.__class__.__name__
    fn_name = 'expr_' + nodetype
    return functions[fn_name]()
      
  def eval_merge_left(phi_nodes):
    for result, (left, _) in phi_nodes.iteritems():
      env[result] = eval_expr(left)
      
  def eval_merge_right(phi_nodes):
    for result, (_, right) in phi_nodes.iteritems():
      env[result] = eval_expr(right)
       
  def eval_stmt(stmt):
    if isinstance(stmt, syntax.Return):
      v = eval_expr(stmt.value)

      raise ReturnValue(v)
    elif isinstance(stmt, syntax.Assign):
      env[stmt.lhs.name] = eval_expr(stmt.rhs)
    elif isinstance(stmt, syntax.If):
      cond_val = eval_expr(stmt.cond)
      if cond_val:
        eval_block(stmt.true)
        eval_merge_left(stmt.merge)
      else:
        eval_block(stmt.false)
        eval_merge_right(stmt.merge)
    elif isinstance(stmt, syntax.While):
      eval_merge_left(stmt.merge_before)
      ran_once = False
      while eval_expr(stmt.cond):
        ran_once = True
        eval_block(stmt.body)
        eval_merge_right(stmt.merge_before)
      if ran_once:
        eval_merge_right(stmt.merge_after)
      else:
        eval_merge_left(stmt.merge_after)
    else: 
      raise RuntimeError("Not implemented: %s" % stmt)
    
  def eval_block(stmts):
    #print "eval", stmts 
    for stmt in stmts:
      eval_stmt(stmt)

  try:
    eval_block(fn.body)
   
  except ReturnValue as r:
    return r.value 
  except:
    raise
  
def run(python_fn, args, type_specialization = False):
  untyped  = ast_conversion.translate_function_value(python_fn)
  # should eventually roll this up into something cleaner, since 
  # top-level functions are really acting like closures over their
  # global dependencies 
  global_args = [python_fn.func_globals[n] for n in untyped.nonlocals]
  all_args = global_args + list(args)
  if not type_specialization:
    return eval_fn(untyped, *all_args) 
  else:
    import type_analysis
    import ptype 
    input_types = map(ptype.type_of_value, all_args)
    typed = type_analysis.specialize(untyped, input_types)
    return eval_fn(typed, *all_args)