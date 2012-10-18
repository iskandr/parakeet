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
 
 
class Args:
  def __init__(self, *args, **kwds):
    self.positional = args
    self.kwds = kwds     

  def bind(self, actuals):
    env = {}
    if hasattr(actuals, 'kwds'):
      kwds = actuals.kwds 
      
    if hasattr(actuals, 'positional'):
      positional = actuals.positional 
    else:
      # assume that we've just been given a list 
      positional = list(actuals)
    
    for k,v in kwds.iteritems():
      env[k] = v
      
    for k,v in self.kwds.iteritems():
      if k not in env:
        env[k] = v
    
    remaining_formals = [arg for arg in self.positional if is_tuple(arg) or name(arg) not in env]
    assert len(positional) == len(remaining_formals), \
      "Unexpected actual arguments (%s) for positional formals (%s)" % (positional, remaining_formals)
    
    match_list(remaining_formals, positional, env)
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
  
  def transform(self, name_fn, tuple_fn = tuple, extract_name = True,
                  kwd_key_fn = None, kwd_value_fn = None):
    positional = transform_list(self.positional, name_fn, tuple_fn, extract_name)
    kwds = {}
    for (k,v) in self.kwds.iteritems():
      old_key = name(k) if extract_name else k
      new_key = kwd_key_fn(old_key) if kwd_key_fn else name_fn(old_key)
      new_value = kwd_value_fn(v) if kwd_value_fn else v
      kwds[new_key] = new_value
    return Args(positional, kwds)
  