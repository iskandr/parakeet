from tree import TreeLike

class Stmt(TreeLike):
  pass 

class Assign(Stmt):
  _members = ['lhs', 'rhs']
  
class Return(Stmt):
  _members = ['value'] 

class If(Stmt):
  _members = ['cond', 'true', 'false', 'merge']
  
class While(Stmt):
  """A loop consists of a header, which runs 
     before each iteration, a condition for 
     continuing, the body of the loop, 
     and optionally (if we're in SSA) merge
     nodes for incoming and outgoing variables
     of the form [(new_var1, (old_var1,old_var2)]
   """
  _members = ['header', 'cond', 'body', 'phi_before', 'phi_after']
  
class Expr(TreeLike):
  pass

class Adverb(Expr):
  pass

class Unop(Expr):
  def __init__(self, op, arg, nonlocal = set([])):
    self.op = op
    self.arg = arg 
    self.nonlocal = nonlocal 
    
class Binop(Expr):
  def __init__(self, op, left, right, nonlocal = set([])):
    self.op = op
    self.left = left
    self.right = right
    self.nonlocal = nonlocal


class Const(Expr):
  def __init__(self, value, t = None):
    self.value = value
    self.type = t

class Var(Expr):
  def __init__(self, name, t = None):
    self.name = name
    self.type = t

class Tuple(Expr):
  def __init__(self, elts, t = None):
    self.elts = elts 
    self.type = t 

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
