
from ..ndtypes import make_closure_type
from expr import Expr
from formal_args import FormalArgs 
from stmt import block_to_str


class UntypedFn(Expr):
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

  #def __str__(self):
  #  return "Fn(%s)" % self.name

  def __repr__(self):
    return "def %s(%s):%s" % (self.name, self.args, block_to_str(self.body))

  def __hash__(self):
    return hash(self.name)

  def node_init(self):
    assert isinstance(self.name, str), \
        "Expected string for fn name, got %s" % self.name
    assert isinstance(self.args, FormalArgs), \
        "Expected arguments to fn to be FormalArgs object, got %s" % self.args
    assert isinstance(self.body, list), \
        "Expected body of fn to be list of statements, got " + str(self.body)

    self.type = make_closure_type(self, ())
    self.registry[self.name] = self

  def python_nonlocals(self):
    if self.python_refs:
      return [ref.deref() for ref in self.python_refs]
    else:
      return []

  def children(self):
    return ()
