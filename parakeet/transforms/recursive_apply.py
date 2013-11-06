from transform import Transform

class RecursiveApply(Transform):  
  def transform_TypedFn(self, expr):
    
    if self.fn.created_by is not None:
      result =  self.fn.created_by.apply(expr)
      return result 
    else:
      # at the very least, apply high level optimizations
      import pipeline
      return pipeline.high_level_optimizations.apply(expr)