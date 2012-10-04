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
  _members = ['cond', 'body', 'merge_before', 'merge_after']
  
class Expr(TreeLike):
  # the type field is initialized to None for untyped syntax nodes
  # but should be set once code gets specialized 
  _members = ['type']

class Adverb(Expr):
  pass


class Const(Expr):
  _members = ['value']
  def __repr__(self):
    return repr(self.value)
  
  def __str__(self):
    return repr(self)
  
class Var(Expr):
  _members = ['name']
  
  def __repr__(self):
    if hasattr(self, 'type'):
      return "var(%s, type=%s)" % (self.name, self.type)
    else:
      return "var(%s)" % self.name 
  
  def __str__(self):
    return self.name 
  
#class Prim(Expr):
#  """Lift primitive to the value level by creating a PrimClosure"""
#  _members = ['value']

class Cast(Expr):
  # inherits the member 'type' from Expr, but for Cast nodes it is mandatory
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
  
class PrimCall(Expr):
  """
  Call a primitive function, the "prim" field should be a 
  prims.Prim object
  """
  _members = ['prim', 'args']
  
  def __repr__(self):
    return "primcall[%s](%s)" % (self.prim, ", ".join(map(str, self.args)))
  
  def __str__(self):
    return repr(self)
  
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
  _members = ['name',  'args', 'body', 'nonlocals']
  
class TypedFn(TreeLike):
  """The body of a TypedFn should contain Expr nodes
  which have been extended with a 'type' attribute
  """
  _members = ['name', 'args', 'body', 'input_types', 'return_type', 'type_env']
      
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
