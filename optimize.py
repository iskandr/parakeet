import syntax 

from simplify import Simplify
from dead_code_elim import DCE
from inline import Inliner
import transform   
from fusion import Fusion 
import config 

def build_pipeline():
  pipeline = [Simplify, DCE]
  def add(t):
    pipeline.append(t)
    if config.opt_cleanup_after_transforms:
      pipeline.append(Simplify)
      pipeline.append(DCE)

  if config.opt_inline:
    add(Inliner)
  
  if config.opt_fusion:
    add(Fusion)
  return pipeline 
    
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
    opt = transform.apply_pipeline(fn, build_pipeline())
    _optimized_cache[fn.name] = opt
    return opt 