import syntax 

from simplify import Simplify
from inline import Inliner, contains_calls  


# map names of unoptimized typed functions to 
# names of optimized 
_optimized_cache = {}
def optimize(fn, copy = False):
  if isinstance(fn, syntax.Fn):
    raise RuntimeError("Can't optimize untyped functions")
  elif isinstance(fn, str):
    assert fn in syntax.TypedFn.registry, \
      "Unknown typed function: " + str(fn)
      
    fn = syntax.TypedFn.registry[fn]
  else:
    assert isinstance(fn, syntax.TypedFn)
      
  if fn.name in _optimized_cache:
    return _optimized_cache[fn.name]
  else:
    opt = Simplify(fn).apply(copy = True)
    if contains_calls(fn):
      inliner = Inliner(opt)
      opt = inliner.apply(copy=False)
    opt =  Simplify(opt).apply(copy=False)
    _optimized_cache[fn.name] = opt
    return opt 