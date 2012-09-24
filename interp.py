import syntax



class ReturnValue(Exception):
  def __init__(self, value):
    self.value = value 

class NotImplemented(Exception):
  pass 


def eval_fn(fn, *args):
  
  #assert False, (fn, args)
  env = dict(zip(fn.args, args)) 
  
  
  def eval_expr(expr):
    def expr_Const():
      return expr.value
    def expr_Binop():
      return expr.op.fn(eval_expr(expr.left), eval_expr(expr.right)) 
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