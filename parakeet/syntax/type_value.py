from .. ndtypes import TypeValueT 
from expr import Expr 


class TypeValue(Expr):
  """
  Value materialization of a type 
  """
  
  _members = ['type_value']
  
  def node_init(self):
    if self.type is None:
      self.type = TypeValueT(self.type_value)
    assert isinstance(self.type, TypeValueT)
    assert self.type.type 
    