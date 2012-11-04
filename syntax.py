from node import Node

class Stmt(Node):
  pass

def block_to_str(stmts):
  body_str = '\n' + '\n'.join([str(stmt) for stmt in stmts])
  return body_str.replace('\n', '\n    ')

def phi_nodes_to_str(phi_nodes):
  parts = ["%s <- phi(%s, %s)" %
           (var, left, right) for (var, (left, right)) in phi_nodes.items()]
  whole = "\n" + "\n".join(parts)
  # add tabs
  return whole.replace("\n", "\n    ")

class Assign(Stmt):
  _members = ['lhs', 'rhs']

  def __str__(self):
    if hasattr(self.lhs, 'type') and self.lhs.type:
      return "%s : %s = %s" % (self.lhs, self.lhs.type, self.rhs)
    else:
      return "%s = %s" % (self.lhs, self.rhs)

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
  _members = ['cond', 'body', 'merge']

  def __repr__(self):
    return "while %s:\n  (merge)%s\n  (body)%s\n" %\
           (self.cond, phi_nodes_to_str(self.merge),  block_to_str(self.body))

  def __str__(self):
    return repr(self)

class Expr(Node):
  _members = ['type']

class Const(Expr):
  _members = ['value']

  def __repr__(self):
    if self.type:
      return "%s : %s" % (self.value, self.type)
    else:
      return str(self.value)
    
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
    if hasattr(self, 'type'):
      return "%s : %s" % (self.name, self.type)
    else:
      return self.name

class Attribute(Expr):
  _members = ['value', 'name']
  
  def __str__(self):
    if self.type:
      return "%s.%s : %s" % (self.value, self.name, self.type)
    else:
      return "%s.%s" % (self.value, self.name)

class Index(Expr):
  _members = ['value', 'index']
  
  def __str__(self):
    if self.type:
      return "%s[%s] : %s" % (self.value, self.index, self.type)
    else:
      return "%s[%s]" % (self.value, self.index)
    
class Tuple(Expr):
  _members = ['elts']
  
  def __str__(self):
    return ", ".join([str(e) for e in self.elts])

class Array(Expr):
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

class Slice(Expr):
  _members = ['lower', 'upper', 'step']

class PrimCall(Expr):
  """
  Call a primitive function, the "prim" field should be a
  prims.Prim object
  """
  _members = ['prim', 'args']

  def __repr__(self):
    if self.prim.symbol:
      if len(self.args) == 1:
        return "%s %s" % (self.prim.symbol)
      else:
        assert len(self.args) == 2
        return "%s %s %s" % (self.args[0], self.prim.symbol, self.args[1])
    else:
      return "prim<%s>(%s)" % (self.prim.name, ", ".join(map(str, self.args)))

  def __str__(self):
    return repr(self)

############################################################################
#
#  Array Operators: It's not scalable to keep adding first-order operators
#  at the syntactic level, so eventually we'll need some more extensible 
#  way to describe the type/shape/compilation semantics of array operators
#
#############################################################################

class Ravel(Expr):
  # given an array, return its data in 1D form 
  _members = ['array']

class ConstArray(Expr):
  _members = ['shape', 'value']

class ConstArrayLike(Expr):
  """
  Create an array with the same shape as the first arg, but with all values 
  set to the second arg
  """
  _members = ['array', 'value']

class Fn(Node):
  """
  Function definition. 
  A top-level function can have a references to python values from its enclosing
  scope, which are stored in the 'python_refs' field. 
  
  A nested function, on the other hand, might refer to some variables from 
  its enclosing Parakeet scope, whose original names are stored in
  'parakeet_nonlocals'
  """
  _members = ['name', 'args', 'body', 'python_refs', 'parakeet_nonlocals']
  
  def __str__(self):
    return "def %s(%s):%s" % (self.name, self.args, block_to_str(self.body))
  def node_init(self):
    
    assert isinstance(self.name, str), \
      "Expected string for fn name, got %s" % self.name
    import args 
    assert isinstance(self.args, args.Args), \
      "Expected arguments to fn to be Args object, got %s" % self.args
    assert isinstance(self.body, list), \
      "Expected body of fn to be list of statements, got " + str(self.body)
  
  def python_nonlocals(self):
    if self.python_refs:
      return [ref.deref() for ref in self.python_refs]
    else:
      return []

##################################################################################
#
#  Constructs below here are only used in the typed representation 
#
##################################################################################

class TupleProj(Expr):
  _members = ['tuple', 'index']
  
  def __str__(self):
    return "%s[%d]" % (self.tuple, self.index) 

class Cast(Expr):
  # inherits the member 'type' from Expr, but for Cast nodes it is mandatory
  _members = ['value']

class Call(Expr):
  """
  Call a function directly, without having to create/invoke a closure
  """
  _members = ['fn', 'args']

class Struct(Expr):
  """
  Eventually all non-scalar data should be transformed to be created
  with this syntax node, signifying explicit struct allocation
  """
  _members = ['args']

class Alloc(Expr):
  """
  Allocates a block of data, returns a pointer
  """
  _members = ['elt_type', 'count']

class TypedFn(Node):
  """The body of a TypedFn should contain Expr nodes
  which have been extended with a 'type' attribute
  """
  _members = ['name', 'args', 'body', 'input_types', 'return_type', 'type_env']

  def __repr__(self):
    args_str = ', '.join([str(slot) for slot in self.args.arg_slots])

    return "function %s(%s):%s" % (self.name, args_str, block_to_str(self.body))

  def __str__(self):
    return repr(self)
