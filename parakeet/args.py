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
    positional = tuple(positional)
    self.positional = positional
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

  def __eq__(self, other):
    if len(self.positional) != len(other.positional) or \
       len(self.keywords) != len(other.keywords) or \
       self.starargs != other.starargs or \
       any(t1 != t2 for (t1, t2) in zip(self.positional, other.positional)):
      return False
    for (k,v) in self.keywords.iteritems():
      if k not in other.keywords:
        return False
      if other.keywords[k] != v:
        return False
    return True

  def __str__(self):
    arg_strings = []
    for p in self.positional:
      arg_strings.append(str(p))
    for (k,v) in self.keywords.items():
      arg_strings.append("%s = %s" % (k,v))
    if self.starargs:
      arg_strings.append("*" + str(self.starargs))
    return ", ".join(arg_strings)

  def __hash__(self):
    kwd_tuple = tuple(self.keywords.items())
    full_tuple = self.positional + kwd_tuple + (self.starargs,)
    return hash(full_tuple)

  def __iter__(self):
    return combine_iters(iter(self.positional),
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
    self.nonlocals = self.nonlocals + tuple(localized_names)

    for (k,p) in self.positions.items():
      if p > n_old_nonlocals:
        self.positions[k] = p + total_nonlocals
    for (i, k) in enumerate(localized_names):
      self.positions[k] = n_old_nonlocals + i

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
      strs.append("*" + str(self.starargs))
    if self.nonlocals:
      strs.append("nonlocals = (%s)" % ", ".join(self.nonlocals))
    return ", ".join(strs)

  def __repr__(self):
    return "Args(positional = %s, defaults=%s, starargs = %s, nonlocal = %s)"% \
           (map(repr, self.positional),
            map(repr, self.defaults.items()),
            self.nonlocals,
            self.starargs)

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
      assert len(extra) == 0, "Too many args: %s" % (extra,)
    return env

  def linearize_values(self, actuals, keyword_fn = None, tuple_elts_fn = iter):
    if isinstance(actuals, (list, tuple)):
      actuals = ActualArgs(actuals)

    positional_values = actuals.positional

    if actuals.starargs:
      starargs_elts = tuple(tuple_elts_fn(actuals.starargs))
      positional_values = positional_values + starargs_elts

    keyword_values = actuals.keywords

    n = self.n_args
    result = [None] * n
    bound = [False] * n

    def assign(i, v):
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
    arg_slots = self.nonlocals + tuple(self.positional)
    missing_args = [arg_slots[i] for i in xrange(n) if not bound[i]]
    assert len(missing_args) == 0, "Missing args: %s" % (missing_args,)
    return result, extra

  def linearize_without_defaults(self, actuals, tuple_elts_fn = tuple):
    linear_args, extra = \
        self.linearize_values(actuals, tuple_elts_fn = tuple_elts_fn,
                              keyword_fn = lambda k, v: None)
    return [x for x in (linear_args + extra) if x is not None]

  def transform(self, rename_fn = lambda x: x, keyword_value_fn = None):
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
