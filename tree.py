# stolen from Pow-wow 

import copy
import logging

def dict_like(value): return isinstance(value, dict)
def tree_like(value): return isinstance(value, TreeLike)
def list_like(value): return hasattr(value, '__iter__')

def flatten(lst):
  res = []
  for elem in lst:
    if list_like(elem): res.extend(flatten(elem))
    else: res.append(elem)
  return res


class BindingMap(dict):
  def __setitem__(self, k, v):
    if k in self:
      raise TypeError, \
        'Duplicate binding for name: %s (assigning %s).  Previously: %s' % \
        (k, v, self[k])
    return dict.__setitem__(self, k, v)
  
  def copy(self):
    return BindingMap(self.items())
  
  def merge(self):
    raise NotImplementedError

class TreeLike(object):
  def __init__(self, *args, **kw):
    self.parent = None
    
    if len(args) > len(self._members):
      raise Exception('Too many arguments for ' + self.__class__.__name__ + 
                      '.  Expected: ' + str(self._members))
      
    for i in range(len(args), len(self._members)):
      arg = self._members[i]
      if not arg in kw:
        logging.fatal("Missing initializer for %s.%s",  
          self.node_type(), self._members[i])

    for field in self._members:
      setattr(self, field, None)

    for field, a in zip(self._members, args):
      setattr(self, field, a)

    for k, v in kw.items():
      if not k in self._members:
        logging.warn('Keyword argument %s not recognized.', k)
      setattr(self, k, v)
      
    self.finalize_init()
        
  def ancestor(self, filter_fn = lambda n: True):
    'Return the first ancestor of this node matching filter_fn'
    n = self.parent
    while n is not None:
#      logging.info('Parent: %s -> %s', self.node_type(), n.node_type())
      if filter_fn(n):
        return n
      n = n.parent
    raise Exception, 'Missing ancestor??'
    
  def expand_children(self, filter_fn=lambda n: True, depth=-1):
    if depth == 0:
      return
    
    if filter_fn(self):
      yield self
  
    for v in self.children():  
      if tree_like(v): 
        yield v.expand_children(filter_fn, depth - 1)
      elif dict_like(v):
        for kk, vv in v.iteritems():
          if tree_like(v[kk]): yield v[kk].expand_children(filter_fn, depth - 1)
      elif list_like(v):
        for i in range(len(v)):
          if tree_like(v[i]): yield v[i].expand_children(filter_fn, depth - 1) 
    
  def finalize_init(self):
    pass

  def node_type(self):
    return '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
  
  def path(self):
    p = []
    n = self
    while n is not None:
      repr = '%s' % n.__class__.__name__
      if hasattr(n, 'name'):
        repr += '(%s)' % n.name 
      p.append(repr)
      n = n.parent
    return '.'.join(reversed(p))

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
        v = getattr(self, k)
        yield k, v 
    
    return dict(_())

  def children(self):
    out = []
    for _, v in self.child_dict().items():
      out.append(v)
    return out
  
  def transform(self, filter_fn, replace_fn):
    self.mark_children()
    for k in self._members:
      v = getattr(self, k)
      if tree_like(v):
        v.transform(filter_fn, replace_fn)
        if filter_fn(v): setattr(self, k, replace_fn(v))
      elif dict_like(v):
        for kk, vv in v.iteritems():
          if filter_fn(vv): v[kk] = replace_fn(vv)
          if tree_like(v[kk]): v[kk].transform(filter_fn, replace_fn)
      elif list_like(v):
        for i in range(len(v)):
          if filter_fn(v[i]): v[i] = replace_fn(v[i])
          if tree_like(v[i]): v[i].transform(filter_fn, replace_fn) 
    
  
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
      if dict_like(v):
        for kk, vv in v.iteritems():
          rv += "%s.%s: %s\n" % (k, kk, vv)
      elif list_like(v):
        for elem in v:
          rv += '%s: %s\n' % (k, elem)
      else:
        rv += '%s: %s\n' % (k, v)
    rv = rv.strip()
    return rv.replace('\n', '\n  |')
  
def find_all(tree, node_type):
  #logging.info('Looking for %s', node_type)    
  return flatten(tree.expand_children(filter_fn=lambda n: isinstance(n, node_type)))
