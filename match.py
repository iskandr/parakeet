import syntax 

def match(pattern, val, env):
  """
  Given a left-hand-side of tuples & vars, 
  a right-hand-side of tuples & types, 
  traverse the tuple structure recursively and
  put the matched variable names in an environment
  """
  if isinstance(pattern, syntax.Var):
    env[pattern.name] = val
  elif isinstance(pattern, syntax.Tuple):
    pat_elts = pattern.elts
    val_elts = val.elts 
    assert len(pat_elts) == len(val_elts), \
      "Mismatch between expected and given number of values"
    for (pi , vi) in zip(pat_elts, val_elts):
      match(pi, vi, env)
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
    if isinstance(p, str):
      match(syntax.Var(p), v, env)
    else:
      match(p, v, env)
  return env  
#
#def flatten_expr_vars(expr):
#  result = set([])
#  def _flatten_list(exprs):
#    for expr in exprs:
#      _flatten(expr)
#      
#  def _flatten(expr):
#    if isinstance(expr, syntax.Var):
#      result.add(expr.name)
#    else:
#      _flatten_list(expr.children())
#  return result       