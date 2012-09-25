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
  _members = ['op', 'value']
  
class Binop(Expr):
  _members = ['op', 'left', 'right']
  

class Const(Expr):
  _members = ['value']
  def __repr__(self):
    return repr(self.value)
  
class Var(Expr):
  _members = ['name']
  
  def __repr__(self):
    return self.name 
  
class Prim(Expr):
  _members = ['value']

class Tuple(Expr):
  _members = ['elts']

class Closure(Expr):
  """
  Create a closure which points to a global fn 
  with a list of partial args
  """
  _members = ['fn', 'args']
  
class Invoke(Expr):
  """
  Invoke a closure with extra args 
  """
  _members = ['closure', 'args']
  

class Call(Expr):
  """
  Call a function directly--- the first argument
  should be a global fn name and not a closure
  """
  _members = ['fn', 'args']
  
class Fn(TreeLike):
  """
  Function definition
  """
  _members = ['name',  'args', 'body']
      
class Traversal:
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
