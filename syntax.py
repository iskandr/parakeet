

class Stmt:
  pass 

class Assign(Stmt):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs

class If(stmt):
  def __init__(self, cond, true, false, merge=None):
    self.cond = cond
    self.true = true
    self.false = false
    # only gets used after conversion to SSA
    self.merge = merge 
    

class While(stmt):
  """A loop consists of a header, which runs 
     before each iteration, a condition for 
     continuing, the body of the loop, 
     and optionally (if we're in SSA) merge
     nodes for incoming and outgoing variables
     of the form [(new_var1, (old_var1,old_var2)]
   """
  def __init__(self, 
    header, cond_expr, body, merge_before = None, merge_after = None):
    self.header 
    self.cond_expr = cond_expr
    self.body = body
    self.merge_before = merge_before
    self.merge_after = merge_after 
 
class Expr:
  pass

class Adverb(Expr):
  pass

class Op(Expr):
  def __init__(self, op, left, right, type = None):
    self.op = op
    self.left = left
    self.right = right
    self.type = type


class Const(Expr):
  def __init__(self, value, type = None):
    self.value = value
    self.type = type

class Var(Expr):
  def __init__(self, id, type = None):
    self.id = id
    self.type = type

class Tuple(Expr):
  def __init__(self, elts, type = None):
    self.elts = elts 
    self.type = type 

class Fn:
  def __init__(self, body, arg_names, defaults, global_vars, types = None):
    self.body = body
    self.args = arg_names
    self.kwds = defaults
    self.global_vars = global_vars
    self.types = types 

    
class SyntaxTraversal:
  def visit_block(self, block, *args, **kwds):
    for stmt in block:
      self.visit_stmt(stmt, *args, **kwds)

  def visit_stmt(self, stmt, *args, **kwds):
    method_name = "stmt_" + stmt.__class__.__name__
    method = getattr(self, method_name)
    method(stmt, *args, **kwds)

  def visit_expr(self, expr, *args, **kwds):
    method_name = "expr_" + expr.__class__.__name__
    method = getattr(self, method_name)
    return method(expr, *args, **kwds)
