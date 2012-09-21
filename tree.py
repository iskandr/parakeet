import copy
import logging


def flatten(lst):
  res = []
  for elem in lst:
    if hasattr(elem, '__iter__'): res.extend(flatten(elem))
    else: res.append(elem)
  return res

class TreeLike(object):
  def __init__(self, *args, **kw):
    if len(args) > len(self._members):
      raise Exception('Too many arguments for ' + self.__class__.__name__ + 
                      '.  Expected: ' + str(self._members))
      
    for i in range(len(args), len(self._members)):
      arg = self._members[i]
      if not arg in kw:
        logging.debug("Missing initializer for %s.%s", 
          self.node_type(), self._members[i])

    for field in self._members:
      setattr(self, field, None)

    for field, a in zip(self._members, args):
      setattr(self, field, a)

    for k, v in kw.items():
      setattr(self, k, v)

  def node_type(self):
    return '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

  def copy(self):
    kw = {}
    for k, v in self.child_dict().items():
      if isinstance(v, TreeLike): v = v.copy()
      else: v = copy.deepcopy(v)
      kw[k] = v
    return self.__class__(**kw)

  def child_dict(self):
    def _():
      for k in self._members:
        yield k, getattr(self, k)
    
    return dict(_())

  def children(self):
    out = []
    for _, v in self.child_dict().items():
      out.append(v)
    return out

  def __cmp__(self, o):
    if not isinstance(o, self.__class__):
      return 1
    return cmp(self.children(), o.children())
  
  def __hash__(self):
    h = 0x123123
    for c in flatten(self.children()):
      h ^= hash(c)
    return h

  def __repr__(self):
    rv = self.node_type() + ':\n'
    for k, v in self.child_dict().items():
      if hasattr(v, '__iter__'):
        for elem in v:
          rv += '%s: %s\n' % (k, elem)
      else:
        rv += '%s: %s\n' % (k, v)
    
    return rv.replace('\n', '\n  |')
  
def find_all(tree, node_type):
  def _(n):
    if not isinstance(n, TreeLike):
      return
    
    for c in flatten(n.children()):
      yield _(c)
    
    if isinstance(n, node_type): 
      yield n
    
      
  return flatten(_(tree))