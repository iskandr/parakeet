import copy
import logging

import sys
FunctionType = sys.modules.get('types').FunctionType

def dict_like(value): return isinstance(value, dict)
def tree_like(value): return isinstance(value, TreeLike)
def list_like(value): return (hasattr(value, '__iter__') 
                              and not dict_like(value) 
                              and not isinstance(value, tuple))

def flatten(lst):
  res = []
  for elem in lst:
    if list_like(elem): res.extend(flatten(elem))
    else: res.append(elem)
  return res


class ScopedMap(object):
  def __init__(self, parent=None):
    self.parent = parent
    self.locals = {}
    self.referenced = {}
    
  def __iter__(self):
    return iter(dict(self.items()))
  
  def keys(self):
    return [k for k, _ in self.items()]
  
  def tree(self):
    m = self
    r = []
    while m:
      r.append(m)
      m = m.parent
    r.reverse()
    return r
  
  def items(self):
    combined = {}
    all_dicts = self.tree()
    for d in all_dicts:
      combined.update(d.locals)
    return combined.items()
  
  def get(self, k, defval):
    if k in self:
      return self[k]
    return defval
  
  def __setitem__(self, k, v):
    assert isinstance(k, str), 'Non-string key in binding map? %s' % k
    # check parent scopes first
    p = self.parent
    while p:
      if k in p.locals: 
        p.locals[k] = v
        return
      p = p.parent
    self.locals[k] = v
  
  def __contains__(self, k):
    if k in self.locals: return True
    if self.parent: return k in self.parent
    return False
  
  def __getitem__(self, k):
    self.referenced[k] = 1
    if k in self.locals: return self.locals[k]
    if self.parent: return self.parent[k]
    raise KeyError, 'Missing key %s' % k
  
  def __repr__(self):
    return 'ScopedMap(%s)' % dict(self.items())

  
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
  
class TreeLike(object):
  def __init__(self, *args, **kw):
    self.parent = None
    if len(args) > len(all_members(self)):
      raise Exception('Too many arguments for ' + self.__class__.__name__ + 
                      '.  Expected: ' + str(all_members(self)))
    
    #for i in range(len(args), len(all_members(self))):
      #arg = all_members(self)[i]
      #assert arg in kw, \
      #  "Missing initializer for %s.%s" % (self.node_type(), all_members(self)[i])

    for field in all_members(self):
      setattr(self, field, None)

    for field, a in zip(all_members(self), args):
      setattr(self, field, a)

    for k, v in kw.items():
      if not k in all_members(self):
        logging.warn('Keyword argument %s not recognized for %s: %s', k, self.node_type(), all_members(self))
      setattr(self, k, v)
      
    self.finalize_init()
    
  def mark_children(self):
    'Mark this node as the parent of any child nodes.'
    for v in flatten(self.expand_children(depth=1)):
      if not tree_like(v):
        continue
      
      if id(v) != id(self) and id(v.parent) != id(self):
        v.parent = self
#        v.mark_children()
        #logging.info('%s -> %s', self.node_type(), v.node_type())
        
  def ancestor(self, filter_fn = lambda n: True):
    'Return the first ancestor of this node matching filter_fn'
    assert isinstance(filter_fn, FunctionType)
    n = self.parent
    while n is not None:
#      logging.info('Parent: %s -> %s', self.node_type(), n.node_type())
      if filter_fn(n):
        return n
      n = n.parent
    raise Exception, 'Missing ancestor??'
    
  def expand_children(self, filter_fn=lambda n: True, depth=-1):
    if filter_fn(self):
      yield self
      
    if depth == 0:
      return
    
    for v in self.children():  
      if tree_like(v): 
        yield v.expand_children(filter_fn, depth - 1)
      elif dict_like(v):
        for kk, vv in v.iteritems():
          if tree_like(v[kk]): yield v[kk].expand_children(filter_fn, depth - 1)
      elif list_like(v):
        for vv in v:
          if tree_like(vv): yield vv.expand_children(filter_fn, depth - 1) 
    
  def finalize_init(self):
    pass

  def node_type(self):
    return self.__class__.__name__
  
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
    d = {}
    for k in all_members(self):
      d[k] = getattr(self, k)
    return d
  
  def children(self):
    return [v for (_, v) in self.child_dict().iteritems()]
  
  def transform(self, filter_fn, replace_fn):
    self.mark_children()
    for k in all_members(self):
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
    mykids = self.children()
    yourkids = o.children()
    return cmp(mykids, yourkids)
  
  def repr(self, only_once):
    def _(node):
      if tree_like(node):
        return node.repr(only_once)
      return repr(node)
    
    if id(self) in only_once: return only_once[id(self)]
    only_once[id(self)] = '<circular reference>'
    
    rv = self.node_type() + ':\n'
    for k, v in self.child_dict().items():
      if dict_like(v):
        for kk, vv in v.iteritems(): rv += '%s.%s : %s\n' % (k, kk, _(vv))
      elif list_like(v):
        for elem in v: rv += '%s: %s\n' % (k, _(elem))
      else:
        rv += '%s: %s\n' % (k, _(v))
    rv = rv.strip()
    only_once[id(self)] = rv.replace('\n', '\n  |')
    return only_once[id(self)]
  
  def __str__(self):
    return self.repr({})
  
  def __repr__(self):
    return self.repr({})
  
def find_all(tree, node_type):
  #logging.info('Looking for %s', node_type)    
  return flatten(tree.expand_children(filter_fn=lambda n: isinstance(n, node_type)))
