import syntax 
import function_registry

import transform
from simplify import Simplify
from inline import Inliner

pipeline = [Simplify, Inliner] 

# map names of unoptimized typed functions to 
# names of optimized 
already_optimized = set([])
def optimize(fn):
  if isinstance(fn, syntax.Fn):
    raise RuntimeError("Can't optimize untyped functions")
  if isinstance(fn, syntax.TypedFn):
    fn_name = fn.name
  else:
    assert isinstance(fn, str)
    fn_name = fn
  fundef = function_registry.typed_functions[fn_name]
  if fn_name in already_optimized:
    return fundef 
  else:
    opt = transform.apply_pipeline(fundef, pipeline)
    already_optimized.add(fn_name)
    return opt 