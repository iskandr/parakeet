
# Copied from Russel's TreeLike but siginificantly 
# stripped down to fit in my underpowered brain 

#import logging 

__member_cache = {}
def all_members(tree):
  'Walk through classes in mro order, accumulating member names.'
  klass = tree.__class__
  if klass in __member_cache: return __member_cache[klass]
    
  m = []
  for c in klass.mro():
    m.extend(getattr(c, '_members', []))
  __member_cache[klass] = m
  return m

  
class Node(object):
  def __init__(self, *args, **kw):
    self.parent = None
    if len(args) > len(all_members(self)):
      raise Exception('Too many arguments for ' + self.__class__.__name__ + 
                      '.  Expected: ' + str(all_members(self)))
    
    #for i in range(len(args), len(all_members(self))):
      #arg = all_members(self)[i]
      #if not arg in kw:
      #  logging.debug("Missing initializer for %s.%s", self.node_type(), all_members(self)[i])
      
    for field in all_members(self):
      setattr(self, field, None)

    for field, a in zip(all_members(self), args):
      setattr(self, field, a)

    for k, v in kw.items():
      if not k in all_members(self):
        raise Exception('Keyword argument %s not recognized for %s: %s', k, self.node_type(), all_members(self))
      setattr(self, k, v)
     
    for C in reversed(self.__class__.mro()):
      if 'finalize_init' in C.__dict__:
        C.finalize_init(self)

  def node_type(self):
    return self.__class__.__name__
  
  def __str__(self):
    members = ["%s = %s" % (m, getattr(self, m)) for m in all_members(self)]
    return "%s(%s)" % (self.node_type(), ", ".join(members))
  
  def __repr__(self):
    return self.__str__()