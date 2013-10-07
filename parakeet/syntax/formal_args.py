from .. import names 
from actual_args import ActualArgs



class UnexpectedKeyword(Exception):
  def __init__(self, keyword_name, fn_name = None):
    self.keyword_name = keyword_name
    self.fn_name = fn_name
  
  def __str__(self):
    if self.fn_name:
      return "Encountered unexpected keyword '%s' in fn %s" % (self.keyword_name, self.fn_name)
    return "Encountered unexpected keyword %s" % self.keyword_name
  
  def __repr__(self):
    return str(self)

class TooManyArgsError(Exception):
  def __init__(self, extra_args, fn_name = None):
    self.extra_args = extra_args
    self.fn_name = fn_name 
  
  def __str__(self):
    if self.fn_name:
      return "Too many args (%s) in call to %s" % (self.extra_args, self.fn_name)
    else:
      return "Too many args: %s" % (self.extra_args,)
    
  
class MissingArg(object):
  pass 
# placeholder object 
missing_arg = MissingArg()


class MissingArgsError(Exception):
  def __init__(self, missing_arg_names, fn_name = None, file_name = None, line_no = None):
    self.missing_arg_names = [names.original(name) for name in missing_arg_names]
    
    self.fn_name = fn_name 
    self.file_name = file_name 
    self.line_no = line_no 
    
  def __str__(self):
    if len(self.missing_arg_names) > 1:
      err_str = "Missing args %s" % ", ".join(["'%s'" % name for name in self.missing_arg_names])
    else:
      err_str = "Missing arg '%s'" % self.missing_arg_names[0]
    
    if self.fn_name is not None:
      err_str += " in call to function '%s'" % self.fn_name
      
    if self.file_name is not None:
      err_str += " in file %s" % self.file_name 
    
    if self.line_no is not None:
      err_str += " on line %d" % self.line_no
    return err_str

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
        s += " = %s" % (self.defaults[local_name],)
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
    elif len(extra) > 0:
      raise TooManyArgsError(extra)
      
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
      assert not bound[i], "%s appears twice in arguments" % self.positional[i]
      bound[i] = True

    if len(positional_values) > n:
      extra = list(positional_values[n:])
      positional_values = positional_values[:n]
    else:
      extra = []

    for (i,p) in enumerate(positional_values):
      assign(i, p)

    for (k,v) in keyword_values.iteritems():
      if k not in self.local_names:
        raise UnexpectedKeyword(k)
      local_name = self.local_names[k]
      assign(self.positions[local_name], v)

    for  (local_name, v) in self.defaults.iteritems():
      i = self.positions[local_name]
      if not bound[i]:
        assign(i, keyword_fn(local_name, v) if keyword_fn else v)
    arg_slots = self.nonlocals + tuple(self.positional)
    missing_args = [arg_slots[i] for i in xrange(n) if not bound[i]]
    if len(missing_args) > 0:
      raise MissingArgsError(missing_args)
    return result, extra
  

  def linearize_without_defaults(self, actuals, tuple_elts_fn = tuple):
    linear_args, extra = \
        self.linearize_values(actuals, tuple_elts_fn = tuple_elts_fn,
                              keyword_fn = lambda k, v: missing_arg)
    return [x for x in (linear_args + extra) if x is not missing_arg]

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
