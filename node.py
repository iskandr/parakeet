
# Copied from Russel's TreeLike but siginificantly 
# stripped down to fit in my underpowered brain 
 
import copy 
from itertools import izip 


_members_cache = {}  
_mro_cache = {}
_reversed_mro_cache = {}
  

class Node(object):
  
  @classmethod
  def get_mro(klass):
    class_name = klass.__name__ 
    if class_name in _mro_cache:
      return _mro_cache[class_name]
    else:
      mro = klass.mro()
      rev_mro = list(reversed(mro))
      _mro_cache[class_name] = mro
      _reversed_mro_cache[class_name] = rev_mro
      return mro 
    
  @classmethod
  def get_reverse_mro(klass):
    class_name = klass.__name__
    if class_name in _reversed_mro_cache:
      return _reversed_mro_cache[class_name]
    else:
      mro = klass.mro()
      rev_mro = list(reversed(mro))
      _mro_cache[class_name] = mro
      _reversed_mro_cache[class_name] = rev_mro
      return rev_mro 
      
      
  @classmethod
  def members(klass):
    """
    Walk through classes in mro order, accumulating member names.
    """
    class_name = klass.__name__
    if class_name  in _members_cache:
      return _members_cache[class_name]
    
    m = []
    for c in klass.get_mro():
      curr_members = getattr(c, '_members', []) 
      for name in curr_members:
        if name not in m:
          m.append(name)  
    _members_cache[class_name] = m
    return m
  
  def iteritems(self):
    for k in self.members():
      yield (k, getattr(self, k, None))
      
  def itervalues(self):
    for (_,v) in self.iteritems():
      yield v 
  
  def items(self):
    [(k,getattr(self,k)) for k in self.members()]
  
  def __init__(self, *args, **kw):
    members = self.members()
    n_args = len(args)
    n_members = len(members)
    class_name = self.__class__.__name__ 
    self_dict = self.__dict__
    if n_args == n_members:
      assert len(kw) == 0
      for (k,v) in zip(members,args):
        self_dict[k] = v
    elif n_args < n_members:
      for field in members:
        self_dict[field] = kw.get(field)
      
      for field, value in izip(members, args):
        self_dict[field] = value
        
      for (k,v) in kw.iteritems():
        assert k in members, \
          "Keyword argument '%s' not recognized for %s: %s" % \
          (k, self.node_type(), members)
    else:
      raise Exception('Too many arguments for %s, expected %s' % \
                      (class_name, members))
       
    # it's more common to not define a node initializer, 
    # so add an extra check to avoid having to always
    # traverse the full class hierarchy 
    if hasattr(self, 'node_init'):
      for C in _reversed_mro_cache[class_name]:
        if 'node_init' in C.__dict__:
          C.node_init(self)

  def __hash__(self):
    # print "Warning: __hash__ not implemented for %s" % self
    hash_values = []
    for m in self.members():
      v = getattr(self, m)
      if isinstance(v, (list, tuple)):
        v = tuple(v)
      hash_values.append(v)
    return hash(tuple(hash_values))
  
  def eq_members(self, other):
    for (k,v) in self.iteritems():
      if not hasattr(other, k):
        return False
      if getattr(other, k) != v:
        return False
    return True 
  
  def __eq__(self, other):
    return other.__class__ is  self.__class__ and self.eq_members(other)

  def __ne__(self, other):
    return not self == other 
  
  @classmethod
  def node_type(cls):
    return cls.__name__
  
  def clone(self, **kwds):
    cloned = copy.deepcopy(self)
    for (k,v) in kwds.values():
      setattr(cloned, k, v)
    return cloned 
    
  def __str__(self):
    member_strings = []
    for (k,v) in self.iteritems():
      member_strings.append("%s = %s" % (k, v)) 
    return "%s(%s)" % (self.node_type(), ", ".join(member_strings))
  
  def __repr__(self):
    return self.__str__()