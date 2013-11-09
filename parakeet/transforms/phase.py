from .. import config

from .. syntax import TypedFn
from clone_function import CloneFunction
from recursive_apply import RecursiveApply
from transform import Transform 

name_stack = []
def apply_transforms(fn, transforms, 
                       cleanup = [], 
                       phase_name = None, 
                       transform_history = None):
  if len(transforms) == 0:
    return fn 
  if phase_name: name_stack.append("{" + phase_name + " :: " + fn.name +  "}")
  for T in transforms:
    t = T() if type(T) == type else T
    
    if isinstance(t, Transform):
      name_stack.append(str(t))
      if config.print_transform_names:
        print "-- %s" % ("->".join(name_stack),)
      
    elif isinstance(t, Phase) and t.should_skip(fn) and not t.depends_on:
      continue 

    fn = t.apply(fn)

    assert fn is not None, "%s transformed fn into None" % T

    if isinstance(t, Transform): name_stack.pop()

    if len(cleanup) > 0:
      fn = apply_transforms(fn, cleanup, [], phase_name = "cleanup")
    
    if transform_history is not None:
      transform_history.add(T)
  
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
               name = None, 
               recursive = True):
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
    self.recursive = recursive 

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
  
  def __call__(self, fn, run_dependencies = True):
    return self.apply(fn, run_dependencies)

  def should_skip(self, fn):
    if self.config_param is not None and getattr(config, self.config_param) == False:
      return True
   
    if self.memoize and  self in fn.transform_history:
      return True
    
    if self.run_if:
      return not self.run_if(fn)
    
    return False 
    
  def is_cached(self, fn):
    return fn.cache_key in self.cache 
  
  def needs_cleanup(self, fn):
    return not (self.should_skip(fn) or self.is_cached(fn)) 
    
  def apply(self, fn, run_dependencies = True):
    
    original_key = fn.cache_key
    if original_key in self.cache:
      return self.cache[original_key]

    if self.depends_on and run_dependencies:
      fn = apply_transforms(fn, self.depends_on)
    
    
    if self.copy:
      fn = CloneFunction(parent_transform = self, rename = self.rename).apply(fn)
      if fn.cache_key  in self.cache:
        "Warning: Typed function %s (key = %s) already registered, encountered while cloning before %s" % \
        (fn.name, fn.cache_key, self)
    
    if self.recursive:
      fn = RecursiveApply(self).apply(fn)
    
      
    if not self.should_skip(fn):
      fn = apply_transforms(fn, self.transforms, 
                          cleanup = self.cleanup, 
                          phase_name = str(self), 
                          transform_history = fn.transform_history)
    
      if self.post_apply:
        new_fn = self.post_apply(fn)
        if new_fn.__class__ is TypedFn:
          fn = new_fn

    fn.transform_history.add(self)
      
    if self.memoize:
      self.cache[original_key] = fn
      self.cache[fn.cache_key] = fn
    return fn
