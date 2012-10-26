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
    val_elts = val.elts 
    assert len(pat_elts) == len(val_elts), \
      "Mismatch between expected and given number of values"
    match_list(pat_elts, val_elts, env, index_fn)
  elif is_index(pattern):
    index_fn(pattern, val, env)
      
  else:
    raise RuntimeError("Unexpected pattern %s %s : %s" % (pattern.__class__.__name__, pattern, val) )    

def match_list(arg_patterns, vals, env = None, index_fn = match_nothing):
  if env is None:
    env = {}
  nargs = len(arg_patterns)
  nvals = len(vals)
  assert nargs == nvals, \
    "Mismatch between %d args and %d input types" % (nargs, nvals)
  for (p,v) in zip(arg_patterns, vals):
    match(p, v, env, index_fn)
  return env  


#def iter_collect(pattern, fn, env, index_fn = match_nothing):
#  if has_name(pattern):
#    n = name(pattern)
#    env[n] = fn(n)
#  else:
#    assert is_tuple(pattern)
#    iter_collect_list(tuple_elts(pattern), fn, env)
#
#def iter_collect_list(patterns, fn, env):
#  for p in patterns:
#    iter_collect(p, fn, env)
  
def transform_nothing(x):
  return x
  
def transform(pat, atom_fn, extract_name = True, tuple_fn = tuple, index_fn = transform_nothing):
  if is_tuple(pat):
    new_elts = transform_list(tuple_elts(pat), atom_fn, extract_name, tuple_fn, index_fn)
    return tuple_fn(new_elts)
  elif is_index(pat):
    return transform_nothing(pat)
  elif extract_name:
    assert has_name(pat)
    return atom_fn(name(pat))
  else:
    return atom_fn(pat)
   
def transform_list(pats, atom_fn,  extract_name = True, tuple_fn = tuple, index_fn = transform_nothing):
  return [transform(p, atom_fn,  extract_name, tuple_fn, index_fn) for p in pats]

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
 
  
class Args:
  def __init__(self, positional, defaults = OrderedDict()):
    assert isinstance(positional, (list, tuple))
    assert isinstance(defaults, OrderedDict)
    self.positional = positional
    self.defaults = defaults
     
    self.arg_slots = list(positional) + defaults.keys()
    self.positions = {}
    for (i, p) in enumerate(self.arg_slots):
      self.positions[p] = i
         

  def __str__(self):
    return "Args(positional = %s, defaults=%s)" % (self.positional, self.defaults.items())
  

  def __iter__(self):
    return iter(self.arg_slots)

  def bind(self, actuals, actual_kwds = {}, default_fn = None):
    """
    Like combine_with_actuals but returns a dictionary
    """
    env = {}
    values = self.linearize_values(actuals, actual_kwds, default_fn)
    for (formal, actual) in zip(self.arg_slots, values):
      match(formal, actual, env)
    return env 
  
  def linearize_values(self, positional_values, keyword_values = {}, default_fn = None):

    n = len(self.arg_slots)
    result = [None] * n
    bound = [False] * n
    def assign(i, v):
      result[i] = v
      assert not bound[i], "%s appears twice in arguments" % self.arg_slots[i]
      bound[i] = True  
      
    for (i,p) in enumerate(positional_values):
      assign(i, p)
      
    for (k,v) in keyword_values.iteritems():
      assert k in self.positions, "Unknown keyword %s" % k
      assign(self.positions[k], v)
      
    for  (k, v) in self.defaults.iteritems():
      i = self.positions[k]
      if not bound[i]:
        assign(i, default_fn(v) if default_fn else v)
    assert all(bound), "Missing args: %s" % [self.arg_slots[i] for i in xrange(n) if not bound[i]]
    return result 

  
  def transform(self, name_fn, tuple_fn = tuple, extract_name = True, keyword_value_fn = None):
    
    
    positional = transform_list(self.positional, name_fn,  extract_name, tuple_fn)
    defaults = OrderedDict()
    for (k,v) in self.defaults.iteritems():
      old_key = name(k) if extract_name else k
      new_key = name_fn(old_key)
      new_value = keyword_value_fn(v) if keyword_value_fn else v
      defaults[new_key] = new_value
    return Args(positional, defaults)
  
  def fresh_copy(self):
    import names 
    
    def name_fn(arg_var):
      new_name = names.refresh(arg_var.name)
      return syntax.Var(name = new_name, type = arg_var.type)
    
    import syntax_helpers 
    import tuple_type 
    def tuple_fn(elts):
      elt_types = syntax_helpers.get_types(elts)
      if any([t is None for t in elt_types]):
        tuple_t = None
      else:
        tuple_t = tuple_type.make_tuple_type(elt_types)
      return syntax.Tuple(elts, type = tuple_t)
    return self.transform(name_fn, tuple_fn, extract_name = False)
    
  