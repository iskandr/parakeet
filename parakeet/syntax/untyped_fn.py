
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
    
  registry = {}

  def __init__(self, name, args, body, 
               python_refs = None, 
               parakeet_nonlocals = None, 
               doc_string = None, 
               source_info = None):
    assert isinstance(name, str), "Expected string for fn name, got %s" % name
    self.name = name 
    
    assert isinstance(args, FormalArgs), \
        "Expected arguments to fn to be FormalArgs object, got %s" % self.args
    self.args = args 
    
    assert isinstance(body, list), \
        "Expected body of fn to be list of statements, got " + str(body)
    self.body = body 
    
    self.python_refs = python_refs 
    self.parakeet_nonlocals = parakeet_nonlocals 
    
    self.type = make_closure_type(self, ())
    
    self.source_info = source_info 
    self.registry[self.name] = self

    
  
  def __repr__(self):
    return "def %s(%s):%s" % (self.name, self.args, block_to_str(self.body))
  
  def __str__(self):
    return repr(self)
  
  def __hash__(self):
    return hash(self.name)

  def python_nonlocals(self):
    if self.python_refs:
      return [ref.deref() for ref in self.python_refs]
    else:
      return []

  def children(self):
    return ()
