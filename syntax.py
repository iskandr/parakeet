import core_types
import args 
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

  def __str__(self):
    return "Return %s" % self.value

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
    if self.type and not isinstance(self.type, core_types.NoneT):
      return "%s : %s" % (self.value, self.type)
    else:
      return str(self.value)

  def __str__(self):
    return repr(self)

class Var(Expr):
  _members = ['name']

  def __repr__(self):
    if hasattr(self, 'type'):
      return "%s : %s" % (self.name, self.type)
    else:
      return "%s" % self.name

  def __str__(self):
    return self.name

class Attribute(Expr):
  _members = ['value', 'name']

  def __str__(self):
    return "attr(%s, '%s')" % (self.value, self.name)

class Index(Expr):
  _members = ['value', 'index']

  def __str__(self):
    return "%s[%s]" % (self.value, self.index)

class Tuple(Expr):
  _members = ['elts']

  def __str__(self):
    return ", ".join([str(e) for e in self.elts])

  def __iter__(self):
    return iter(self.elts)

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
        
  def __str__(self):
    return "%s(%s)" % (self.closure, self.args)
   
  def __repr__(self):
    return str(self)

class Slice(Expr):
  _members = ['start', 'stop', 'step']

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

"""
class Unpack(Expr):
  # Unpack a varargs tuple into an argument list
  _members = ['value']

  def __str__(self):
    return "*%s" % self.value

  def __repr__(self):
    return str(self)
"""
############################################################################
#
#  Array Operators: It's not scalable to keep adding first-order operators
#  at the syntactic level, so eventually we'll need some more extensible
#  way to describe the type/shape/compilation semantics of array operators
#
#############################################################################
"""
class Ravel(Expr):
  # given an array, return its data in 1D form
  _members = ['array']
"""

class ConstArray(Expr):
  _members = ['shape', 'value']

class ConstArrayLike(Expr):
  """
  Create an array with the same shape as the first arg, but with all values
  set to the second arg
  """
  _members = ['array', 'value']

class ArrayView(Expr):
  """
  Create a new view on already allocated underlying data
  """
  _members = ['data', 'shape', 'strides']

class Fn(Expr):
  """
  Function definition.
  A top-level function can have references to python values from its enclosing
  scope, which are stored in the 'python_refs' field.

  A nested function, on the other hand, might refer to some variables from
  its enclosing Parakeet scope, whose original names are stored in
  'parakeet_nonlocals'
  """
  _members = ['name', 'args', 'body', 'python_refs', 'parakeet_nonlocals']

  def __str__(self):
    return "def %s(%s):%s" % (self.name, self.args, block_to_str(self.body))

  def __repr__(self):
    return str(self)

  def node_init(self):
    assert isinstance(self.name, str), \
      "Expected string for fn name, got %s" % self.name

    assert isinstance(self.args, args.FormalArgs), \
      "Expected arguments to fn to be FormalArgs object, got %s" % self.args
    assert isinstance(self.body, list), \
      "Expected body of fn to be list of statements, got " + str(self.body)
    
    import closure_type 
    self.type = closure_type.ClosureT(self.name, ())

  def python_nonlocals(self):
    if self.python_refs:
      return [ref.deref() for ref in self.python_refs]
    else:
      return []

################################################################################
#
#  Constructs below here are only used in the typed representation
#
################################################################################

class TupleProj(Expr):
  _members = ['tuple', 'index']

class ClosureElt(Expr):
  _members = ['closure', 'index']

  def __str__(self):
    return "ClosureElt(%s, %d)" % (self.closure, self.index)

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

  def __str__(self):
    return "Struct(%s) : %s" % \
        (", ".join(str(arg) for arg in self.args), self.type)

class Alloc(Expr):
  """
  Allocates a block of data, returns a pointer
  """
  _members = ['elt_type', 'count']

  def __str__(self):
    return "alloc<%s>[%s] : %s" % (self.elt_type, self.count, self.type)

class IntToPtr(Expr):
  """
  Reinterpret an integer as a pointer to the specified type
  """
  _members = ['value']

class PtrToInt(Expr):
  """
  Convert the address of a pointer into an integer
  """
  _members = ['value']

class TypedFn(Expr):
  """
  The body of a TypedFn should contain Expr nodes
  which have been extended with a 'type' attribute
  """
  _members = ['name',
              'arg_names',
              #'num_varargs',
              'body',
              'input_types',
              'return_type',
              'type_env']

  def node_init(self):
    self.type = core_types.make_fn_type(self.input_types, self.return_type)

  def __repr__(self):
    return "function %s(%s):%s" % \
      (self.name, self.arg_names, block_to_str(self.body))

  def __str__(self):
    return repr(self)
