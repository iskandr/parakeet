from transform import Transform 

class ShapePropagation(Transform):
  """
  a = IndexMap(f, bounds)
  b = a.shape
   
  -becomes-
  
  b = bounds 
  
  """
  
  def transform_IndexMap(self, expr):
    return expr 
  
  def transform_Shape(self, expr):
    return expr 
  