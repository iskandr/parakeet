
# Copied from Russel's TreeLike but siginificantly 
# stripped down to fit in my underpowered brain 
 
import copy 


class Node(object):
  _members_cache = {}  
  
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
    return m
  
  
  def __init__(self, *args, **kw):
    self.parent = None
    members = self.members()
    if len(args) > len(members):
      raise Exception('Too many arguments for ' + self.__class__.__name__ + 
                      '.  Expected: ' + str(members))
    for field in members:
      setattr(self, field, None)

    for field, a in zip(members, args):
      setattr(self, field, a)

    for k, v in kw.items():
      if not k in members:
        raise Exception("Keyword argument '%s' not recognized for %s: %s" % \
                        (k, self.node_type(), members))
      setattr(self, k, v)
     
    for C in reversed(self.__class__.mro()):
      if 'node_init' in C.__dict__:
        C.node_init(self)

  @classmethod
  def node_type(cls):
    return cls.__name__
  
  def clone(self, **kwds):

    cloned = copy.deepcopy(self)
    for (k,v) in kwds.values():
      setattr(cloned, k, v)
    return cloned 
    
  
  def __str__(self):
    members = ["%s = %s" % (m, getattr(self, m)) for m in self.members()]
    return "%s(%s)" % (self.node_type(), ", ".join(members))
  
  def __repr__(self):
    return self.__str__()