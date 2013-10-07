from run_function import specialize

def typed_repr(python_fn, args, optimize = True):
  typed_fn, _ = specialize(python_fn, args)
  if optimize:
    from ..transforms import pipeline
    return pipeline.high_level_optimizations.apply(typed_fn)
  else:
    return typed_fn 
  
  
