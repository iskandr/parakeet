
from expr import Expr 


class DelayUntilTyped(Expr):
  """
  Once the list of values has been annotated with locally inferred types, 
  pass them to the given function to construct a final expression 
  """
  def __init__(self, values, keywords, fn, source_info = None):
    """
    No need for a 'type' argument since the user-supplied function 
    will generate a typed expression to replace this one
    """
    if isinstance(values, list): values = tuple(values)
    elif not isinstance(values, tuple): values = (self.values,)
    self.values = values 
    
    if keywords is None: keywords = {}
    self.keywords = keywords 
    
    self.fn = fn 
    self.source_info = source_info 
    
    
  def children(self):
    return tuple(self.values) + tuple(self.keywords.values())  