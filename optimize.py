import syntax 

from simplify import Simplify
from dead_code_elim import DCE
from inline import Inliner
from clone_function import CloneFunction
import transform   
from fusion import Fusion 
pipeline = [CloneFunction, Simplify, DCE, Inliner, Simplify, Fusion, Simplify, DCE] 

# map names of unoptimized typed functions to 
# names of optimized 
_optimized_cache = {}
def optimize(fn):
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
    opt = transform.apply_pipeline(fn, pipeline)
    _optimized_cache[fn.name] = opt
    return opt 