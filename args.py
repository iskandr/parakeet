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


class FormalArgs(object):
  def __init__(self):
    self.n_args = 0
    self.nonlocals = ()
    self.positional = []
     
    self.defaults = {}
    self.starargs = None 
    
    # map visible name to local SSA name
    self.local_names = {}
    # map SSA name to visible (keyword) name
    self.visible_names = {}
    
    # position of each local name in the bound args list 
    self.positions = {}
    
    
    
  def _prepend(self, local_name, visible_name = None):
    self.n_args += 1
    if visible_name:
      self.local_names[visible_name] = local_name
      self.visible_names[local_name] = visible_name
    self.arg_slots = [local_name] + self.arg_slots 
    for k in self.positions:
      self.positions[k] += 1
    self.positions[local_name] = 0
    self.positional = [local_name] + self.positional
  
  def add_positional(self, local_name, visible_name = None):
    self.n_args += 1
    if visible_name:
      self.local_names[visible_name] = local_name
      self.visible_names[local_name] = visible_name
    self.positions[local_name] = len(self.positions)
    self.positional.append(local_name)
  
    
  def prepend_nonlocal_args(self, localized_names):
    n_old_nonlocals = len(self.nonlocals)
    n_new_nonlocals = len(localized_names)
    total_nonlocals = n_old_nonlocals + n_new_nonlocals
    self.n_args += n_new_nonlocals
    self.nonlocals = self.nonlocals  + tuple(localized_names)
    
    for (k,p) in self.positions.items():
      if p < n_old_nonlocals:
        self.positions[k] = p + n_new_nonlocals
      else:
        self.positions[k] = p + total_nonlocals
    
  def __str__(self):
    strs = []
    for local_name in self.positional:
      if local_name in self.visible_names:
        s = "%s{%s}" % (local_name, self.visible_names[local_name])
      else:
        s = local_name
      if local_name in self.defaults:
        s += " = %s" % self.defaults[local_name]
      strs.append(s)
    if self.starargs:
      strs.append("*%s" % self.starargs)
    return ", ".join(strs)
  
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
  
  
  def bind(self, actuals, 
           keyword_fn = None, 
           tuple_elts_fn = iter, 
           starargs_fn = tuple):
    """
    Like combine_with_actuals but returns a dictionary
    """
    env = {}
    values, extra = self.linearize_values(actuals, keyword_fn, tuple_elts_fn)
    
    for (k,v) in zip(self.nonlocals + tuple(self.positional), values):
      env[k] = v
    
    if self.starargs:
      env[self.starargs] = starargs_fn(extra)
    else:
      assert len(extra) == 0, "Too many args: %s" % (extra, )
    return env

  def linearize_values(self, actuals, keyword_fn = None, tuple_elts_fn = iter):
    if isinstance(actuals, (list, tuple)):
      actuals = ActualArgs(actuals)
      
    positional_values =  actuals.positional
    
    if actuals.starargs:
      starargs_elts = tuple(tuple_elts_fn(actuals.starargs))
      positional_values = positional_values + starargs_elts
       
    keyword_values = actuals.keywords 
    
    n = self.n_args
    result = [None] * n
    bound = [False] * n

    def assign(i, v):
      print "-- %d = %s" % (i, v)
      result[i] = v
      assert not bound[i], "%s appears twice in arguments" % self.arg_slots[i]
      bound[i] = True

    if len(positional_values) > n:
      extra = list(positional_values[n:])
      positional_values = positional_values[:n]
    else:
      extra = []

    for (i,p) in enumerate(positional_values):
      assign(i, p)

    for (k,v) in keyword_values.iteritems():
      assert k in self.local_names, \
        "Unknown keyword: %s" % k
      local_name = self.local_names[k]
      assign(self.positions[local_name], v)

    for  (local_name, v) in self.defaults.iteritems():
      i = self.positions[local_name]
      if not bound[i]:
        assign(i, keyword_fn(local_name, v) if keyword_fn else v)

    missing_args = [self.positional[i] for i in xrange(n) if not bound[i]]
    assert len(missing_args) == 0, "Missing args: %s" % (missing_args,)
    return result, extra

  def transform(self, 
                rename_fn = lambda x: x, 
                keyword_value_fn = None):
    
    args = FormalArgs()
    
    args.prepend_nonlocal_args(map(rename_fn, self.nonlocals))
    
    for old_local_name in self.positional:
      new_local_name = rename_fn(old_local_name)
      visible_name = self.visible_names.get(old_local_name)
      args.add_positional(new_local_name, visible_name)
      if old_local_name in self.defaults:
        v = self.defaults[old_local_name]
        if keyword_value_fn:
          v = keyword_value_fn(new_local_name, v)
        args.defaults[new_local_name] = v
    args.starargs = rename_fn(self.starargs) if self.starargs else None
    return args 
