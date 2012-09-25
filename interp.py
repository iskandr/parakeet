import syntax
import ast_conversion 
from global_state import untyped_functions

class ReturnValue(Exception):
  def __init__(self, value):
    self.value = value 

class NotImplemented(Exception):
  pass 


def eval_fn(fn, *args):
  
  env = dict(zip(fn.args, args)) 
  def eval_expr(expr):    
    def expr_Const():
      return expr.value
    def expr_Binop():
      fn = eval_expr(expr.op)
      left = eval_expr(expr.left)
      
      right = eval_expr(expr.right)
      #assert False, (left, right)
      return fn(left, right) 
    def expr_Prim():
      return expr.value.fn
    def expr_Var():
      return env[expr.name]
    def expr_Invoke():
      fn_name, closure_args = eval_expr(expr.closure)
      arg_vals = map(eval_expr, expr.args)
      combined_arg_vals = closure_args + arg_vals 
      fn = untyped_functions[fn_name]
      return eval_fn(fn, *combined_arg_vals)
    def expr_Closure():
      return expr.fn, map(eval_expr, expr.args)
    
    nodetype = expr.__class__.__name__ 
    return locals()['expr_' + nodetype]()
    
  def eval_stmt(stmt):
    if isinstance(stmt, syntax.Return):
      v = eval_expr(stmt.value)
      raise ReturnValue(v)
    elif isinstance(stmt, syntax.Assign):
      env[stmt.lhs] = eval_expr(stmt.rhs)
    else: 
      raise NotImplemented 
  def eval_block(stmts):
    for stmt in stmts:
      eval_stmt(stmt)
  try:
    eval_block(fn.body)
  except ReturnValue as r:
    return r.value 
  except:
    raise
  
def run(python_fn, *args):
  untyped  = ast_conversion.translate_function_value(python_fn)
  return eval_fn(untyped, *args) 