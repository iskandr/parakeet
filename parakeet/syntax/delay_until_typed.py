
from expr import Expr 


class DelayUntilTyped(Expr):
  """
  Once the list of values has been annotated with locally inferred types, 
  pass them to the given function to construct a final expression 
  """
  _members = ['values', 'keywords', 'fn']
 
  def node_init(self):
    if isinstance(self.values, list):
      self.values = tuple(self.values)
    elif not isinstance(self.values, tuple):
      self.values = (self.values,)
    if self.keywords is None:
      self.keywords = {}
    
  def children(self):
    return tuple(self.values) + tuple(self.keywords.values())  