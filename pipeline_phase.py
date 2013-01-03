import config 
from clone_function import CloneFunction
from syntax import TypedFn 
  

def apply_transforms(fn, transforms, cleanup = []):
  for T in transforms:
    t = T() if type(T) == type else T
    fn = t.apply(fn)
    assert fn is not None, "%s transformed fn into None" % T
 
    if cleanup and \
       (not isinstance(t, Phase) or t.cleanup != cleanup):
      fn = apply_transforms(fn, cleanup, [])
  return fn 

class Phase(object):

  def __init__(self, 
                transforms,
                depends_on = [],  
                copy = False, 
                cleanup = [], 
                config_param = None, 
                skip_if = None, 
                rename = False,  
                run_after = None, 
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
    
    self.skip_if = skip_if 
    
    self.rename = rename 
    self.run_after = run_after 
    self.memoize = memoize 
    self.name = name   
  
    def __str__(self):
      if self.name:
        return self.name 
      else:
        return "Phase(%s)" % (",".join(str(t) for t in self.transforms))
    
  
  def __call__(self, fn, run_dependencies = True):
    return self.apply(fn, run_dependencies)
  
  def apply(self, fn, run_dependencies = True):
    if self.config_param is not None and \
       getattr(config, self.config_param) == False:
      return fn 
    
    original_key = fn.name, fn.copied_by  
    if original_key in self.cache:
      return self.cache[original_key]
    
    if self.depends_on and run_dependencies:
      fn = apply_transforms(fn, self.depends_on)
    
    if self.copy:
      fn = CloneFunction(self.rename).apply(fn)
      fn.copied_by = self 

    if (self.skip_if is None) or (not self.skip_if(fn)):
      fn = apply_transforms(fn, self.transforms, cleanup = self.cleanup)
      
    if self.run_after:
      new_fn = self.run_after(fn)
      if new_fn.__class__ is TypedFn:
        fn = new_fn 

    if self.memoize:
      self.cache[original_key] = fn
      new_key = fn.name, fn.copied_by  
      self.cache[new_key] = fn
    return fn   