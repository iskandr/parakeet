import args
from   args import ActualArgs
import core_types
from  node import Node
from syntax_stmts import * 
from syntax_expr import *
from syntax_array_expr import * 
from syntax_adverbs import * 


class DelayUntilTyped(Expr):
  """
  Once the list of values has been annotated with locally inferred types, 
  pass them to the given function to construct a final expression 
  """
  _members = ['values', 'fn']
 
  def node_init(self):
    if isinstance(self.values, list):
      self.values = tuple(self.values)
    elif not isinstance(self.values, tuple):
      self.values = (self.values,)
    
  def children(self):
    return self.values

class TypeValue(Expr):
  """
  Value materialization of a type 
  """
  
  _members = ['type_value']
  
  def node_init(self):
    if self.type is None:
      self.type = core_types.TypeValueT(self.type_value)
    assert isinstance(self.type, core_types.TypeValueT)
    assert self.type.type 
    
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
    self.type = closure_type.make_closure_type(self, ())
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
              'version',]

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

  def __repr__(self):
    arg_strings = []

    for name in self.arg_names:
      arg_strings.append("%s : %s" % (name, self.type_env.get(name)))

    return "function %s(%s) => %s:%s" % \
           (self.name, ", ".join(arg_strings),
            self.return_type,
            block_to_str(self.body))

  def __str__(self):
    return "TypedFn(%s : %s => %s)" % (self.name, self.input_types, self.return_type)

  def __hash__(self):
    return hash(self.name)

  def children(self):
    return ()
