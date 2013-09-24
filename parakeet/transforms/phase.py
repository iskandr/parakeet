from .. import config

from .. syntax import TypedFn
from clone_function import CloneFunction
from transform import Transform 

name_stack = []
def apply_transforms(fn, transforms, cleanup = [], phase_name = None):
  if phase_name: name_stack.append("{" + phase_name + "}")
  for T in transforms:
    t = T() if type(T) == type else T
    if isinstance(t, Transform): 
      name_stack.append(str(t))
      if config.print_transform_names:
        print "-- Running %s on %s" % ("->".join(name_stack), fn.name)
    elif isinstance(t, Phase) and t.run_if is not None and not t.run_if(fn):
      continue 
    fn = t.apply(fn)
    assert fn is not None, "%s transformed fn into None" % T
  
    if isinstance(t, Transform): name_stack.pop()
    fn = apply_transforms(fn, cleanup, [], phase_name = "cleanup")
  
  if phase_name: name_stack.pop()
  return fn

name_stack = []
class Phase(object):
  def __init__(self,
               transforms,
               depends_on = [],
               copy = False,
               cleanup = [],
               config_param = None,
               run_if = None,
               rename = False,
               post_apply = None,
               memoize = True,
               name = None):
    self.cache = {}
    if not isinstance(transforms, (tuple, list)):
      transforms = [transforms]
    self.transforms = transforms

    if not depends_on:
      depends_on = []
    if not isinstance(depends_on, (tuple, list)):
      depends_on = [depends_on]
    self.depends_on = depends_on

    self.copy = copy

    if not cleanup:
      cleanup = []
    if not isinstance(cleanup, (list,tuple)):
      cleanup = [cleanup]
    self.cleanup = cleanup

    self.config_param = config_param
    self.run_if = run_if
    self.rename = rename
    self.post_apply = post_apply
    self.memoize = memoize
    self.name = name

  def __str__(self):
    if self.name:
      return self.name
    else:
      names = []
      for t in self.transforms:
        if type(t) == type:
          names.append(t.__name__)
        else:
          names.append(str(t))
      if len(names) == 1:
        return names[0]
      else:
        return "Phase(%s)" % (",".join(names))

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return str(self) == str(other)

  def __hash__(self):
    return hash(str(self))
  
  def __call__(self, fn, run_dependencies = True, ignore_config = False):
    return self.apply(fn, run_dependencies, ignore_config)

  def apply(self, fn, run_dependencies = True, ignore_config = False):
    if self.config_param is not None and not ignore_config and \
       getattr(config, self.config_param) == False:
      return fn
    
    if self.memoize and (fn.created_by == self or self in fn.transform_history):
      return fn 
    
    original_key = fn.cache_key
    if original_key in self.cache:
      return self.cache[original_key]

    if self.depends_on and run_dependencies:
      fn = apply_transforms(fn, self.depends_on)
      
    if self.run_if is not None and not self.run_if(fn):
      return fn 
    
    if self.copy:
      fn = CloneFunction(self.rename).apply(fn)
      fn.created_by = self

    fn.transform_history.add(self)
    fn = apply_transforms(fn, self.transforms, cleanup = self.cleanup, phase_name = str(self))

    
    if self.post_apply:
      new_fn = self.post_apply(fn)
      if new_fn.__class__ is TypedFn:
        fn = new_fn

    if self.memoize:
      self.cache[original_key] = fn
      self.cache[fn.cache_key] = fn
    return fn
