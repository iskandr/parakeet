import syntax

from collections import OrderedDict

def has_name(arg):
  return isinstance(arg, str) or isinstance(arg, syntax.Var)

def name(arg):
  return arg if isinstance(arg, str) else arg.name

def is_tuple(arg):
  return hasattr(arg, 'elts') or hasattr(arg, '__iter__')

def tuple_elts(arg):
  try:
    return arg.elts
  except:
    # hope that it has an __iter__ implementation
    return tuple(arg)

def is_index(arg):
  return isinstance(arg, syntax.Index)

def flatten(arg):
  if is_tuple(arg):
    return flatten_list(tuple_elts(arg))
  else:
    return [arg]

def flatten_list(args):
  result = []
  for arg in args:
    result += flatten(arg)
  return result

def match_nothing(pattern, val, env):
  pass

def match(pattern, val, env, index_fn = match_nothing):
  """
  Given a left-hand-side of tuples & vars,
  a right-hand-side of tuples & types,
  traverse the tuple structure recursively and
  put the matched variable names in an environment
  """
  if has_name(pattern):
    env[name(pattern)] = val
  elif is_tuple(pattern):
    pat_elts = tuple_elts(pattern)
    val_elts = tuple_elts(val)
    assert len(pat_elts) == len(val_elts), \
      "Mismatch between expected and given number of values"
    match_list(pat_elts, val_elts, env, index_fn)
  elif is_index(pattern):
    index_fn(pattern, val, env)

  else:
    raise RuntimeError("Unexpected pattern %s %s : %s" %
                       (pattern.__class__.__name__, pattern, val))

def match_list(arg_patterns, vals, env = None, index_fn = match_nothing):
  if env is None:
    env = {}
  nargs = len(arg_patterns)
  nvals = len(vals)
  assert nargs == nvals, \
    "Mismatch between %d args %s and %d inputs %s" % (nargs, arg_patterns,
                                                      nvals, vals)
  for (p,v) in zip(arg_patterns, vals):
    match(p, v, env, index_fn)
  return env

def transform_nothing(x):
  return x

def transform(pat, atom_fn, extract_name = True, tuple_fn = tuple,
              index_fn = transform_nothing):
  if is_tuple(pat):
    new_elts = transform_list(tuple_elts(pat), atom_fn, extract_name, tuple_fn,
                              index_fn)
    return tuple_fn(new_elts)
  elif is_index(pat):
    return transform_nothing(pat)
  elif extract_name:
    assert has_name(pat)
    return atom_fn(name(pat))
  else:
    return atom_fn(pat)

def transform_list(pats, atom_fn,  extract_name = True, tuple_fn = tuple,
                   index_fn = transform_nothing):
  return [transform(p, atom_fn,  extract_name, tuple_fn, index_fn)
          for p in pats]


class CombinedIters:
  def __init__(self, i1, i2):
    self.i1 = i1
    self.i2 = i2
    self.first_iter = True 
    
  def next(self):
    if self.first_iter:
      try:
        return self.i1.next()
      except:
        self.first_iter = False
        return self.i2.next()
    else:
      return self.i2.next()

def combine_iters(*iters):
  assert len(iters) > 0
  curr_iter = iters[0]
  for i in iters[1:]:
    curr_iter = CombinedIters(curr_iter, i)
  return curr_iter 

class NullIter(object):
  def next(self):
    raise StopIteration

def maybe_iter(obj):

  if obj is None:
    return NullIter()
  else:
    return iter(obj)

class ActualArgs(object):
  def __init__(self, positional, keywords = {}, starargs = None):
    self.positional = tuple(positional) 
    self.keywords = keywords 
    self.starargs = starargs 
    
  def transform(self, fn, keyword_name_fn = None, keyword_value_fn = None):
    new_pos = map(fn, self.positional)
    new_keywords = {}
    for (k,v) in self.keywords.iteritems():
      new_name = keyword_name_fn(k) if keyword_name_fn else k 
      new_value = keyword_value_fn(v) if keyword_value_fn else fn(v)
      new_keywords[new_name] = new_value
    new_starargs = fn(self.starargs) if self.starargs else None 
    return ActualArgs(new_pos, new_keywords, new_starargs)
  
  def __str__(self):
    arg_strings = []
    for p in self.positional:
      arg_strings.append(str(p))
    for (k,v) in self.keywords.items():
      arg_strings.append("%s = %s" % (k,v))
    if self.starargs:
      arg_strings.append("*%s" % self.starargs)
    return ", ".join(arg_strings) 
  
  def __hash__(self):
    kwd_tuple = tuple(self.keywords.items())
    return hash(self.positional + kwd_tuple + (self.starargs,))
    
  def __iter__(self):
    return combine_iters(
      iter(self.positional), 
      self.keywords.itervalues(), 
      maybe_iter(self.starargs))
        
  
  def prepend_positional(self, more_args):
    new_pos = tuple(more_args) + self.positional 
    return ActualArgs(new_pos, self.keywords, self.starargs)

import names 

class FormalArgs(object):

  
  def __init__(self, positional = (),
                     defaults = OrderedDict(),
                     nonlocals = (),
                     starargs = None, 
                     local_name_fn = lambda x: x):
    
    assert isinstance(positional, (list, tuple))
    assert isinstance(defaults, OrderedDict)
    
    self.nonlocals = tuple(nonlocals)
    self.positional = tuple(positional)
    self.defaults = defaults
    self.starargs = starargs
    
    self.local_names = {}
    self.arg_slots = []
    self.positions = {}
    
    self.prepend_nonlocal_args(self.nonlocals, local_name_fn)
    
    for x in positional:
      self.add_positional(x, local_name_fn)
    for (k,v) in defaults.iteritems():
      self.add_keyword_arg(k, v, local_name_fn)
    
    if starargs:
      self.add_starargs(starargs, local_name_fn)

    self.visible_names = list(reversed(self.local_names.keys()))
    
  
  def add_positional(self, pos_arg, local_name_fn = lambda x:x):
    assert len(self.defaults) == 0, \
      "Can't add positional args after defaults"
    assert self.starargs is None, \
      "Can't add positional args after starargs" 
    if is_tuple(pos_arg):
      return tuple(self.add_positional(elt, local_name_fn) for elt in pos_arg)
    else:
      visible_name = name(pos_arg) 
      self.positions[visible_name] = len(self.positions)
      local_name = local_name_fn(visible_name)
      self.local_names[visible_name] = local_name
      self.arg_slots.append(local_name)
      return local_name 
  
  def add_keyword_arg(self, key, value, local_name_fn = lambda x:x):
    assert self.starargs is None 
    assert isinstance(name, (str, syntax.Var))
    visible_name = name(key)
    local_name = local_name_fn(visible_name)  
    self.arg_slots.append(local_name)
    self.positions[visible_name] = len(self.positions)
    self.local_names[visible_name] = local_name
    return local_name 
  
  def add_starargs(self, starargs, local_name_fn = lambda x:x):
    assert self.starargs is None, \
      "Function can't have two starargs variables"
    assert isinstance(starargs, (str, syntax.Var))
    visible_name = name(starargs)
    local_name = local_name_fn(visible_name)
    self.local_names[visible_name] = local_name
    self.positions[visible_name] = len(self.positions)
    return local_name 
    
  def prepend_nonlocal_args(self, more_nonlocals, local_name_fn = lambda x:x):
    localized_names = map(local_name_fn, more_nonlocals)
    self.nonlocals = tuple(more_nonlocals) + self.nonlocals 
    self.arg_slots = list(localized_names) + self.arg_slots
    n = len(more_nonlocals)
    for k in self.positions:
      self.positions[k] += n
    for (i,(l,k)) in enumerate(zip(localized_names, more_nonlocals)):
      self.positions[k] = i
      self.local_names[k] = l 
  
  def __str__(self):
    
    def arg_to_str(arg):
      if is_tuple(arg):
        return "(%s)" % ", ".join([arg_to_str(e) for e in tuple_elts(arg)])
      else:
        visible_name = name(arg)
        local_name = self.local_names[visible_name]
        if visible_name == local_name:
          return visible_name
        else:
          return "%s{%s}" % (visible_name, local_name)  
    arg_strings = []
    for pos_arg in self.positional:
      arg_strings.append(arg_to_str(pos_arg))
    for (k,v) in self.defaults.items():
      visible_name = name(k)
      local_name = self.local_names[visible_name]
      arg_strings.append("%s{%s} = %s" % (visible_name, local_name, v))
    if self.starargs:
      visible_name = name(self.starargs)
      local_name = self.local_names[visible_name]
      if visible_name == local_name:
        arg_strings.append("*%s" % visible_name)
      else:
        arg_strings.append("*%s{%s}" % (visible_name, local_name))
    
    arg_strings += \
      ["nonlocals = (%s)" % ", ".join(map(str, self.nonlocals))] \
      if self.nonlocals else []
    return ", ".join(arg_strings)

  def __repr__(self):
    return "Args(positional = %s, defaults=%s, starargs = %s, nonlocal = %s)" % \
      (
       map(repr, self.positional),
       map(repr, self.defaults.items()),
       self.nonlocals,
       self.starargs
      )

  def __iter__(self):
    if self.starargs:
      return iter(self.arg_slots + [self.starargs])
    else:
      return iter(self.arg_slots)

  def keywords(self):
    return map(name, self.defaults.keys())
  
  
  def bind(self, actuals, default_fn = None, starargs_fn = tuple):
    """
    Like combine_with_actuals but returns a dictionary
    """
    env = {}
    values, extra = self.linearize_values(actuals, default_fn)
    for (formal, actual) in zip(self.arg_slots, values):
      match(formal, actual, env)

    if self.starargs:
      env[self.local_names[self.starargs]] = starargs_fn(extra)
    else:
      assert len(extra) == 0, "Too many args: %s" % (extra, )
    return env

  def linearize_values(self, actuals, default_fn = None):
    if isinstance(actuals, (list, tuple)):
      actuals = ActualArgs(actuals)
    positional_values = actuals.positional
    keyword_values = actuals.keywords 
    starargs = actuals.starargs 
    
    
    n = len(self.arg_slots)
    result = [None] * n
    bound = [False] * n

    def assign(i, v):
      result[i] = v
      assert not bound[i], "%s appears twice in arguments" % self.arg_slots[i]
      bound[i] = True

    if len(positional_values) > n:
      extra = positional_values[n:]
      positional_values = positional_values[:n]
    else:
      extra = []

    for (i,p) in enumerate(positional_values):
      assign(i, p)

    for (k,v) in keyword_values.iteritems():
      print self.positions
      assert k in self.positions, "Unknown keyword %s" % k
      assign(self.positions[k], v)

    for  (k, v) in self.defaults.iteritems():
      i = self.positions[k]
      if not bound[i]:
        assign(i, default_fn(k,v) if default_fn else v)

    if starargs:
      unbound_positions = [i for i in xrange(n) if not bound[i]]
      counter = 0 
      for arg in starargs:
        if counter < len(unbound_positions):
          pos = unbound_positions[counter]
          assign(pos, arg)
          counter += 1
        else:
          extra.append(arg)
          
    missing_args = [self.arg_slots[i] for i in xrange(n) if not bound[i]]

    assert len(missing_args) == 0, "Missing args: %s" % (missing_args,)
    return result, extra

  def transform(self, 
                name_fn, 
                tuple_fn = tuple, 
                extract_name = True,
                keyword_value_fn = None, 
                local_name_fn = None):
    
    nonlocals = transform_list(self.nonlocals, name_fn, extract_name, tuple_fn)
    positional = transform_list(self.positional, name_fn,
                                extract_name, tuple_fn)

    starargs = name_fn(self.starargs) if self.starargs else None

    defaults = OrderedDict()
    for (k,v) in self.defaults.iteritems():
      old_key = name(k) if extract_name else k
      new_key = name_fn(old_key)
      new_value = keyword_value_fn(v) if keyword_value_fn else v
      defaults[new_key] = new_value
      
    if local_name_fn is None:
      local_name_fn = lambda visible_name: self.local_names[visible_name]
    return FormalArgs(positional, defaults, nonlocals, starargs, local_name_fn)

  def fresh_copy(self, local_name_fn = names.refresh):
    return FormalArgs(positional = self.positional, 
               defaults = self.defaults, 
               nonlocals = self.nonlocals, 
               starargs = self.starargs, 
               local_name_fn = local_name_fn)
    