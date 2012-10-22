import syntax 

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

def match(pattern, val, env):
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
    val_elts = val.elts 
    assert len(pat_elts) == len(val_elts), \
      "Mismatch between expected and given number of values"
    match_list(pat_elts, val_elts, env)
  else:
    raise RuntimeError("Unexpected pattern %s %s : %s" % (pattern.__class__.__name__, pattern, val) )    

def match_list(arg_patterns, vals, env = None):
  if env is None:
    env = {}
  nargs = len(arg_patterns)
  nvals = len(vals)
  assert nargs == nvals, \
    "Mismatch between %d args and %d input types" % (nargs, nvals)
  for (p,v) in zip(arg_patterns, vals):
    match(p, v, env)
  return env  


def iter_collect(pattern, fn, env):
  if has_name(pattern):
    n = name(pattern)
    env[n] = fn(n)
  else:
    assert is_tuple(pattern)
    iter_collect_list(tuple_elts(pattern), fn, env)

def iter_collect_list(patterns, fn, env):
  for p in patterns:
    iter_collect(p, fn, env)
  
def transform(pat, atom_fn, tuple_fn = tuple, extract_name = True):
  if is_tuple(pat):
    new_elts = transform_list(tuple_elts(pat), atom_fn, tuple_fn, extract_name)
    return tuple_fn(new_elts)
  elif extract_name:
    assert has_name(pat)
    return atom_fn(name(pat))
  else:
    return atom_fn(pat)
   
def transform_list(pats, atom_fn, tuple_fn = tuple, extract_name = True):
  return [transform(p, atom_fn, tuple_fn, extract_name) for p in pats]

def bind(lhs, rhs):
  if isinstance(lhs, Args):
    return lhs.bind(rhs)
  
  if isinstance(lhs, (tuple, list)):
    keys = flatten_list(lhs)
  else:
    keys = flatten(lhs)
    
  if isinstance(rhs, (list, tuple)):
    values = flatten_list(rhs)
  else:
    values = flatten(rhs)
  
  return dict(*zip(keys, values)) 
 
class KeyValueList:
  
  def __init__(self, *elts):
    self.elts = elts
    self.positions = {}
    for (i, (k, _)) in elts:
      self.positions[k] = i
    
  def pos(self, k):
    return self.positions[k]
  
  def __iter__(self):
    return iter(self.keys())
  
  def iteritems(self):
    return iter(self.elts)
  
  def iterkeys(self):
    return (k for (k, _) in self.elts)
  
  def itervalues(self):
    return (v for (_, v) in self.elts)
   
  def keys(self):
    return [k for (k,_) in self.elts]
  
  def values(self):
    return [v for (_, v) in self.elts]
   
  def __contains__(self, k):
    return k in self.keys() 
  
  def __str__(self):
    return str(self.elts)
  
  def __len__(self):
    return len(self.elts)
  
class Args:
  def __init__(self, positional, kwds = KeyValueList()):
    assert isinstance(positional, (list, tuple))
    self.positional = positional
    assert isinstance(kwds, KeyValueList)
    self.kwds = kwds     

  def __str__(self):
    return "Args(positional = %s, kwds=%s)" % (self.positional, self.kwds)
  
  def flatten(self, default_keys = True):
    return flatten(self.positional) + (self.kwds.keys() if default_keys else self.kwds.values()) 
  
  def bind(self, actuals, actual_kwds = None):
    """
    Like combine_with_actuals but returns a dictionary
    """
    env = {}
    for (formal, actual) in self.combine(actuals, actual_kwds):
      match(formal, actual, env)
    return env 
  
  def combine_with_actuals(self, actuals, actual_kwds = None):
    
    if isinstance(actuals, Args):
      assert len(actual_kwds) == 0
      kwds = actuals.kwds
      positional = actuals.positional
    else:
      positional = list(actuals)
    
    if actual_kwds is None:
      kwds = KeyValueList()
    elif isinstance(actual_kwds, (list, tuple)):
      kwds = KeyValueList(actual_kwds)
    
    formals_list = []
    actuals_list = []
    
    n_positional = len(self.positional)
    used = set([])
    for (i, arg) in enumerate(positional):
      if i < n_positional:
        formal = self.positional[i]
      else:
        formal = self.kwds[i - n_positional][0]
      formals_list[i] = formal
      actuals_list[i] = arg 
      used.add(formal)
    
    for (k,v) in kwds.iteritems():
      assert k not in used
      used.add(k)
      self.kwds.pos(k)
      env[k] = v
      
    for (k,v) in self.kwds.iteritems():
      if k not in env:
        env[k] = v 
    return env   
   
  
  def iter_collect(self, name_fn, kwd_item_fn = None):
    """ 
    Apply given function to every arg and accumulate the outputs
    associated with arg names in a dictionary
    """
    env = {}
    iter_collect_list(self.positional, name_fn, env)
    for (k, v) in self.kwds.iteritems():
      k = name(k)
      new_v = kwd_item_fn(k, v) if kwd_item_fn else name_fn(k) 
      env[k] = new_v
    return env
  
  def transform(self, positional_fn, tuple_fn = tuple, extract_name = True,
                  kwd_key_fn = None, kwd_value_fn = None):
    
    positional = transform_list(self.positional, positional_fn, tuple_fn, extract_name)
    kwds = {}
    for (k,v) in self.kwds.iteritems():
      old_key = name(k) if extract_name else k
      new_key = kwd_key_fn(old_key) if kwd_key_fn else positional_fn(old_key)
      new_value = kwd_value_fn(v) if kwd_value_fn else v
      kwds[new_key] = new_value

    return Args(positional, kwds) 
  