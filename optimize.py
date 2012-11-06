import syntax 
import function_registry

import transform
from simplify import Simplify
from inline import Inliner

pipeline = [Simplify] #, Inliner, Simplify] 


# map names of unoptimized typed functions to 
# names of optimized 
_optimized_functions = {}
def optimize(fn):
  if isinstance(fn, syntax.Fn):
    raise RuntimeError("Can't optimize untyped functions")
  if isinstance(fn, syntax.TypedFn):
    fn_name = fn.name
  else:
    assert isinstance(fn, str)
    fn_name = fn
    
  if fn_name in _optimized_functions:
    opt_name = _optimized_functions[fn_name]
    return function_registry.typed_functions[opt_name]
  else:
    assert fn_name in function_registry.typed_functions
    unopt = function_registry.typed_functions[fn_name]
    opt = transform.apply_pipeline(unopt, pipeline)
    _optimized_functions[fn_name] = opt.name
    # also register the optimized function with itself
    # so trying to optimize an already optimized fn
    # acts like the identity 
    _optimized_functions[opt.name] = opt.name
    return opt 