import args
from   args import ActualArgs
import core_types
from   node import Node

class Stmt(Node):
  pass

def block_to_str(stmts):
  body_str = '\n'
  body_str += '\n'.join([str(stmt) for stmt in stmts])

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

class ExprStmt(Stmt):
  """Run an expression without binding any new variables"""

  _members = ['value']

  def __str__(self):
    assert self.value is not None
    return "ExprStmt(%s)" % self.value

class Comment(Stmt):
  _members = ['text']

  def __str__(self):
    s = "#"
    for (i, c) in enumerate(self.text):
      if i % 78 == 0:
        s += "\n# "
      s += c
    s += "\n#"
    return s


class Return(Stmt):
  _members = ['value']

  def __str__(self):
    return "Return %s" % self.value

class If(Stmt):
  _members = ['cond', 'true', 'false', 'merge']

  def __str__(self):
    s = "if %s:" % self.cond
    if (len(self.true) + len(self.false)) > 0:
      s += "%s\n" % block_to_str(self.true)
    else:
      s += "\n"
    if len(self.false) > 0:
      s += "else:%s\n" % block_to_str(self.false)
    if len(self.merge) > 0:
      s += "(merge-if)%s" % phi_nodes_to_str(self.merge)
    return s

  def __repr__(self):
    return str(self)

class While(Stmt):
  """
  A loop consists of a header, which runs before each iteration, a condition for
  continuing, the body of the loop, and optionally (if we're in SSA) merge nodes
  for incoming and outgoing variables of the form
  [(new_var1, (old_var1,old_var2)]
  """

  _members = ['cond', 'body', 'merge']

  def __repr__(self):
    s = "while %s:\n  "  % self.cond
    if len(self.merge) > 0:
      s += "(header)%s\n  " % phi_nodes_to_str(self.merge)
    if len(self.body) > 0:
      s +=  "(body)%s" % block_to_str(self.body)
    return s

  def __str__(self):
    return repr(self)

class ForLoop(Stmt):
  """
  Having only one loop construct started to become cumbersome, especially now
  that we're playing with loop optimizations.

  So, here we have the stately and ancient for loop.  All hail its glory.
  """

  _members = ['var', 'start', 'stop', 'step', 'body', 'merge']

  def __str__(self):
    s = "for %s in range(%s, %s, %s):" % \
      (self.var,
       self.start.short_str(),
       self.stop.short_str(),
       self.step.short_str())

    if self.merge and len(self.merge) > 0:
      s += "\n  (header)%s\n  (body)" % phi_nodes_to_str(self.merge)
    s += block_to_str(self.body)
    return s

class Expr(Node):
  _members = ['type']

  def children(self):
    for v in self.itervalues():
      if v and isinstance(v, Expr):
        yield v
      elif isinstance(v, (list,tuple)):
        for child in v:
          if isinstance(child, Expr):
            yield child

  def short_str(self):
    return str(self)

class Const(Expr):
  _members = ['value']

  def children(self):
    return (self.value,)

  def short_str(self):
    return str(self.value)

  def __repr__(self):
    if self.type and not isinstance(self.type, core_types.NoneT):
      return "%s : %s" % (self.value, self.type)
    else:
      return str(self.value)

  def __str__(self):
    return repr(self)

  def __hash__(self):
    return hash(self.value)

  def __eq__(self, other):
    return other.__class__ is Const and \
           self.value == other.value and \
           self.type == other.type

  def __ne__(self, other):
    return other.__class__ is not Const or \
           self.value != other.value or \
           self.type != other.type

class Var(Expr):
  _members = ['name']

  def short_str(self):
    return self.name

  def __repr__(self):
    if hasattr(self, 'type'):
      return "%s : %s" % (self.name, self.type)
    else:
      return "%s" % self.name

  def __str__(self):
    return self.name

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return other.__class__  is Var and \
           self.name == other.name and \
           self.type == other.type


  def __ne__(self, other):
    return other.__class__ is not Var or \
           self.name != other.name or \
           self.type != other.type
  
  def children(self):
    return ()

class Attribute(Expr):
  _members = ['value', 'name']

  def children(self):
    yield self.value

  def __str__(self):
    return "attr(%s, '%s')" % (self.value, self.name)

  def __hash__(self):
    return hash ((self.value, self.name))

  def __eq__(self, other):
    return other.__class__ is Attribute and \
           self.name == other.name and \
           self.value == other.value

class Index(Expr):
  _members = ['value', 'index']
  
  def __eq__(self, other):
    return other.__class__ is Index and \
           other.value == self.value and \
           other.index == self.index
  
  def __hash__(self):
    return hash((self.value, self.index))
  
  def children(self):
    yield self.value
    yield self.index

  def __str__(self):
    return "%s[%s]" % (self.value, self.index)

class Tuple(Expr):
  _members = ['elts']

  def __str__(self):
    if len(self.elts) > 0:
      return ", ".join([str(e) for e in self.elts])
    else:
      return "()"

  def node_init(self):
    self.elts = tuple(self.elts)

  def children(self):
    return self.elts

  def __hash__(self):
    return hash(self.elts)

class Array(Expr):
  _members = ['elts']

  def node_init(self):
    self.elts = tuple(self.elts)

  def children(self):
    return self.elts

  def __hash__(self):
    return hash(self.elts)

class Closure(Expr):
  """Create a closure which points to a global fn with a list of partial args"""

  _members = ['fn', 'args']

  def __str__(self):
    fn_str = str(self.fn) #self.fn.name if hasattr(self.fn, 'name') else str(self.fn)
    args_str = ",".join(arg.name + ":" + str(arg.type) for arg in self.args)
    return "Closure(fixed_args = {%s}, %s)" % (args_str, fn_str)

  def node_init(self):
    self.args = tuple(self.args)

  def children(self):
    if isinstance(self.fn, Expr):
      yield self.fn
    for arg in self.args:
      yield arg

  def __hash__(self):
    return hash((self.fn, tuple(self.args)))

class Call(Expr):
  _members = ['fn', 'args']

  def __str__(self):
    #if isinstance(self.fn, (Fn, TypedFn)):
    #  fn_name = self.fn.name
    #else:
    fn_name = str(self.fn)
    if isinstance(self.args, ActualArgs):
      arg_str = str(self.args)
    else:
      arg_str = ", ".join(str(arg) for arg in self.args)
    return "%s(%s)" % (fn_name, arg_str)

  def __repr__(self):
    return str(self)

  def children(self):
    yield self.fn
    for arg in self.args:
      yield arg

  def __hash__(self):
    return hash((self.fn, tuple(self.args)))

class Slice(Expr):
  _members = ['start', 'stop', 'step']

  def __str__(self):
    return "slice(%s, %s, %s)"  % \
        (self.start.short_str(),
         self.stop.short_str(),
         self.step.short_str())

  def __repr__(self):
    return str(self)

  def children(self):
    yield self.start
    yield self.stop
    yield self.step

  def __eq__(self, other):
    return other.__class__ is Slice and \
           other.start == self.start and \
           other.stop == self.stop and \
           other.step == self.step
  def __hash__(self):
    return hash((self.start, self.stop, self.step))

class PrimCall(Expr):
  """
  Call a primitive function, the "prim" field should be a prims.Prim object
  """
  _members = ['prim', 'args']
  
  def _arg_str(self, i):
    arg = self.args[i]
    if arg.__class__ is PrimCall:
      return "(%s)" % arg
    else:
      return str(arg)
  
  def __hash__(self):
    self.args = tuple(self.args)
    return hash((self.prim, self.args))
    
  def __eq__(self, other):
    if other.__class__ is not PrimCall:
      return False 
    
    if other.prim != self.prim:
      return False
    
    my_args = self.args
    other_args = other.args
    n = len(my_args)
    if n != len(other_args):
      return False
    
    for (i,arg) in enumerate(my_args):
      other_arg = other_args[i]
      if arg != other_arg:
        return False
    return True
  
  def __ne__(self, other):
    return not self==other
  
  def __repr__(self):
    if self.prim.symbol:
      if len(self.args) == 1:
        return "%s %s" % (self._arg_to_str(0), self.prim.symbol)
      else:
        assert len(self.args) == 2
        return "%s %s %s" % (self._arg_str(0), self.prim.symbol, self._arg_str(1))
    else:
      arg_strings = [self._arg_str(i) for i in xrange(len(self.args))]
      combined = ", ".join(arg_strings)
      return "prim<%s>(%s)" % (self.prim.name, combined)

  def __str__(self):
    return repr(self)

  def node_init(self):
    self.args = tuple(self.args)

  def children(self):
    return self.args

#############################################################################
#
#  Array Operators: It's not scalable to keep adding first-order operators
#  at the syntactic level, so eventually we'll need some more extensible
#  way to describe the type/shape/compilation semantics of array operators
#
#############################################################################

class Len(Expr):
  _members = ['value']

class ConstArray(Expr):
  _members = ['shape', 'value']

class ConstArrayLike(Expr):
  """
  Create an array with the same shape as the first arg, but with all values set
  to the second arg
  """

  _members = ['array', 'value']

class Range(Expr):
  _members = ['start', 'stop', 'step']

class AllocArray(Expr):
  """Allocate an unfilled array of the given shape and type"""
  _members = ['shape']

  def children(self):
    yield self.shape

class ArrayView(Expr):
  """Create a new view on already allocated underlying data"""

  _members = ['data', 'shape', 'strides', 'offset', 'total_elts']

  def children(self):
    yield self.data
    yield self.shape
    yield self.strides
    yield self.offset
    yield self.total_elts

class Fn(Expr):
  """
  Function definition.
  A top-level function can have references to python values from its enclosing
  scope, which are stored in the 'python_refs' field.

  A nested function, on the other hand, might refer to some variables from its
  enclosing Parakeet scope, whose original names are stored in
  'parakeet_nonlocals'
  """

  _members = ['name', 'args', 'body', 'python_refs', 'parakeet_nonlocals']
  registry = {}

  def __str__(self):
    return "Fn(%s)" % self.name
    # return repr(self)

  def __repr__(self):
    return "def %s(%s):%s" % (self.name, self.args, block_to_str(self.body))

  def __hash__(self):
    return hash(self.name)

  def node_init(self):
    assert isinstance(self.name, str), \
        "Expected string for fn name, got %s" % self.name
    assert isinstance(self.args, args.FormalArgs), \
        "Expected arguments to fn to be FormalArgs object, got %s" % self.args
    assert isinstance(self.body, list), \
        "Expected body of fn to be list of statements, got " + str(self.body)

    import closure_type
    self.type = closure_type.make_closure_type(self.name, ())
    self.registry[self.name] = self

  def python_nonlocals(self):
    if self.python_refs:
      return [ref.deref() for ref in self.python_refs]
    else:
      return []

  def children(self):
    return ()

################################################################################
#
#  Constructs below here are only used in the typed representation
#
################################################################################

class TupleProj(Expr):
  _members = ['tuple', 'index']

  def __str__(self):
    return "%s[%d]" % (self.tuple, self.index)

  def children(self):
    return (self.tuple,)

  def __hash__(self):
    return hash((self.tuple, self.index))

class ClosureElt(Expr):
  _members = ['closure', 'index']

  def __str__(self):
    return "ClosureElt(%s, %d)" % (self.closure, self.index)

  def children(self):
    return (self.closure,)

  def __hash__(self):
    return hash((self.closure, self.index))

class Cast(Expr):
  # inherits the member 'type' from Expr, but for Cast nodes it is mandatory
  _members = ['value']

  def __hash__(self):
    return hash(self.value)

class Struct(Expr):
  """
  Eventually all non-scalar data should be transformed to be created with this
  syntax node, signifying explicit struct allocation
  """

  _members = ['args']

  def __str__(self):
    return "Struct(%s) : %s" % \
           (", ".join(str(arg) for arg in self.args), self.type)

  def children(self):
    return self.args

  def __hash__(self):
    self.args = tuple(self.args)
    return hash(self.args)

class Alloc(Expr):
  """Allocates a block of data, returns a pointer"""

  _members = ['elt_type', 'count']

  def __str__(self):
    return "alloc<%s>[%s] : %s" % (self.elt_type, self.count, self.type)

  def children(self):
    return (self.count,)

  def __hash__(self):
    return hash((self.elt_type, self.count))

class TypedFn(Expr):
  """
  The body of a TypedFn should contain Expr nodes which have been extended with
  a 'type' attribute
  """

  _members = ['name',
              'arg_names',
              'body',
              'input_types',
              'return_type',
              'type_env',
              # these last two get filled by
              # transformation/optimizations later
              'copied_by',
              'version',
              'has_tiles',
              'num_tiles',
              'dl_tile_estimates',
              'ml_tile_estimates',
              'autotuned_tile_sizes']

  registry = {}
  max_version = {}
  def next_version(self, name):
    n = self.max_version[name] + 1
    self.max_version[name] = n
    return n

  def node_init(self):
    assert isinstance(self.body, list), \
        "Invalid body for typed function: %s" % (self.body,)
    assert isinstance(self.arg_names, (list, tuple)), \
        "Invalid typed function arguments: %s" % (self.arg_names,)
    assert isinstance(self.name, str), \
        "Invalid typed function name: %s" % (self.name,)

    if isinstance(self.input_types, list):
      self.input_types = tuple(self.input_types)

    assert isinstance(self.input_types, tuple), \
        "Invalid input types: %s" % (self.input_types,)
    assert isinstance(self.return_type, core_types.Type), \
        "Invalid return type: %s" % (self.return_type,)
    assert isinstance(self.type_env, dict), \
        "Invalid type environment: %s" % (self.type_env,)

    self.type = core_types.make_fn_type(self.input_types, self.return_type)

    if self.version is None:
      self.version = 0
      self.max_version[self.name] = 0

    registry_key = (self.name, self.version)
    assert registry_key not in self.registry, \
        "Typed function %s version %s already registered" % \
        (self.name, self.version)
    self.registry[registry_key] = self

    if self.has_tiles is None:
      self.has_tiles = False
    if self.num_tiles is None:
      self.num_tiles = 0
    if self.dl_tile_estimates is None:
      self.dl_tile_estimates = []
    if self.ml_tile_estimates is None:
      self.ml_tile_estimates = []

  def __repr__(self):
    arg_strings = []

    for name in self.arg_names:
      arg_strings.append("%s : %s" % (name, self.type_env.get(name)))

    return "function %s(%s) => %s:%s" % \
           (self.name, ", ".join(arg_strings),
            self.return_type,
            block_to_str(self.body))

  def __str__(self):
    return self.__repr__()

  def __hash__(self):
    return hash(self.name)

  def children(self):
    return ()
