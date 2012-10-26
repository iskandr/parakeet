import transform

class LowerAdverbs(transform.Transform):
  
  def transform_Adverb(self, expr):
    pass 
  

def lower_adverbs(fn):
  return transform.cached_apply(LowerAdverbs, fn)
  