from transform import Transform

class IndexMapElimination(Transform):
  """
  a = IndexMap(f, bounds)
  ... a[i] ... 
  
  -becomes-
  
  ... f(i) ... 
  """
  def transform_IndexMap(self, expr):
    return expr 