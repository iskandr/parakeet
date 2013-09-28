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
    new_pos = [fn(pos_arg) for pos_arg in self.positional]
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

  def __repr__(self):
    return str(self)

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
