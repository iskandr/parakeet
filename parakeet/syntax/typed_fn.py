
from .. ndtypes import make_fn_type, Type 


from expr import Expr 
from stmt import block_to_str

################################################################################
#
#  Constructs below here are only used in the typed representation
#
################################################################################
class TransformHistory(object):
  """
  Sequence of transforms which have been applied to a function, 
  used for caching to avoid repeating a transformation
  """
  
  def __init__(self, transforms = None):
    self.transforms = () if transforms is None else transforms 
    self.transform_set = frozenset(self.transforms) 
    self._hash = hash(self.transform_set)
    
  def __contains__(self, T):
    return T in self.transform_set
  
  def add(self, T):
    self.transforms = self.transforms + (T,)
    self.transform_set = self.transform_set.union(frozenset([T]))
    self._hash = hash(self.transform_set)
  
  @property
  def cache_key(self):
    return self.transform_set
  
  def __hash__(self):
    return self._hash
  
  def __eq__(self, other):
    return self.transform_set == other.transform_set
  
  def __ne__(self, other):
    return self.transform_set != other.transform_set 
  
  def __str__(self):
    return str(self.transforms)
  
  def copy(self):
    return TransformHistory()

class TypedFn(Expr):
  """
  The body of a TypedFn should contain Expr nodes which have been extended with
  a 'type' attribute
  """
  
  def __init__(self, name, arg_names, body, 
                input_types, return_type,
                type_env, 
                created_by = None,
                transform_history = None,  
                source_info = None):
    
    assert isinstance(name, str), "Invalid typed function name: %s" % (name,)
    self.name = name 
    
    assert isinstance(arg_names, (list, tuple)), "Invalid typed function arguments: %s" % (arg_names,)
    self.arg_names = arg_names 

    assert isinstance(input_types, (list, tuple)), "Invalid input types: %s" % (input_types,)
    self.input_types = tuple(input_types)    
    
    assert isinstance(return_type, Type), "Invalid return type: %s" % (return_type,)
    self.return_type = return_type 
    
    assert isinstance(body, list), "Invalid body for typed function: %s" % (body,)
    self.body = body 
    
    assert isinstance(type_env, dict), "Invalid type environment: %s" % (type_env,)
    self.type_env = type_env 

    self.type = make_fn_type(self.input_types, self.return_type)
    
    self.created_by = created_by 
    
    if transform_history is None: 
      transform_history = TransformHistory()
    self.transform_history = transform_history
    
    self.source_info = source_info 


  @property 
  def cache_key(self):
    return self.name, self.created_by, self.transform_history
  
  @property
  def version(self):
    return self.transform_history.cache_key 
  
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
