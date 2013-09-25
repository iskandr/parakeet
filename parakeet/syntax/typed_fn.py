
from .. ndtypes import make_fn_type, Type 


from expr import Expr 
from stmt import block_to_str

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
              'created_by',
              'transform_history', 
             ]
  
  @property 
  def cache_key(self):
    return self.name, self.created_by, self.version
  
  @property
  def version(self):
    return frozenset(self.transform_history)
  
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
    assert isinstance(self.return_type, Type), \
        "Invalid return type: %s" % (self.return_type,)
    assert isinstance(self.type_env, dict), \
        "Invalid type environment: %s" % (self.type_env,)

    self.type = make_fn_type(self.input_types, self.return_type)
    
    if self.transform_history is None:
      self.transform_history = set([])



  def __repr__(self):
    arg_strings = []

    for name in self.arg_names:
      arg_strings.append("%s : %s" % (name, self.type_env.get(name)))

    return "function %s(%s) => %s:%s" % \
           (self.name, ", ".join(arg_strings),
            self.return_type,
            block_to_str(self.body))

  def __str__(self):
    return repr(self)
    # return "TypedFn(%s : %s => %s)" % (self.name, self.input_types, self.return_type)

  def __hash__(self):
    return hash(self.name)

  def children(self):
    return ()
