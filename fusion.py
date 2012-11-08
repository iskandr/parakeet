import transform

class Fusion(transform.Transform):
  def transform_Map(self, expr):
    return expr 
  
  def transform_Reduce(self, expr):
    return expr
  
  def transform_AllPairs(self, expr):
    return expr
  
  def transform_Scan(self, expr):
    return expr 
  
  def transform_Assign(self, stmt):
    return stmt 