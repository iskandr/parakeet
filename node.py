
# Copied from Russel's TreeLike but siginificantly 
# stripped down to fit in my underpowered brain 
 
import copy 


class Node(object):
  _members_cache = {}  
  _members_set_cache = {}
  
  
  @classmethod
  def members(klass):
    'Walk through classes in mro order, accumulating member names.'
    if klass in klass._members_cache:
      return klass._members_cache[klass]
    
    m = []
    for c in klass.mro():
      curr_members = getattr(c, '_members', []) 
      for name in curr_members:
        if name not in m:
          m.append(name)  
    klass._members_cache[klass] = m
    klass._members_set_cache[klass] = set(m)
    return m
  
  
  def iteritems(self):
    for k in self.members():
      yield (k, getattr(self, k))
      
  def itervalues(self):
    for (_,v) in self.iteritems():
      yield v 
  
  
  def items(self):
    [(k,getattr(self,k)) for k in self.members()]
  
  def __init__(self, *args, **kw):
    members = self.members()
    if len(args) > len(members):
      raise Exception('Too many arguments for ' + self.__class__.__name__ + 
                      '.  Expected: ' + str(members))
    
    for field in members:
      value = kw.get(field, None)
      setattr(self, field, value)
      
    for field, value in zip(members, args):
      setattr(self, field, value)

    klass = self.__class__ 
    members_set = self._members_set_cache[klass]
    for k in kw.iterkeys():
      assert k in members_set, \
          "Keyword argument '%s' not recognized for %s: %s" % \
          (k, self.node_type(), members)
     
    for C in reversed(klass.mro()):
      if 'node_init' in C.__dict__:
        C.node_init(self)

  def __hash__(self):
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
    return other.__class__ ==  self.__class__ and self.eq_members(other)

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