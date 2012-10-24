
# Copied from Russel's TreeLike but siginificantly 
# stripped down to fit in my underpowered brain 

#import logging 
import copy 
    


class Node(object):
  _members_cache = {}  
  
  @property
  def members(self):
    'Walk through classes in mro order, accumulating member names.'
    klass = self.__class__
    if klass in self._members_cache:
      return self._members_cache[klass]
    
    m = []
    for c in klass.mro():
      m.extend(getattr(c, '_members', []))
    self._members_cache[klass] = m
    return m
  
  
  def __init__(self, *args, **kw):
    self.parent = None
    members = self.members 
    if len(args) > len(members):
      raise Exception('Too many arguments for ' + self.__class__.__name__ + 
                      '.  Expected: ' + str(members))
    
    #for i in range(len(args), len(all_members(self))):
      #arg = all_members(self)[i]
      #if not arg in kw:
      #  logging.debug("Missing initializer for %s.%s", self.node_type(), all_members(self)[i])
      
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

  def node_type(self):
    return self.__class__.__name__
  
  def clone(self, **kwds):

    cloned = copy.deepcopy(self)
    for (k,v) in kwds.values():
      setattr(cloned, k, v)
    return cloned 
    
  
  def __str__(self):
    members = ["%s = %s" % (m, getattr(self, m)) for m in self.members]
    return "%s(%s)" % (self.node_type(), ", ".join(members))
  
  def __repr__(self):
    return self.__str__()