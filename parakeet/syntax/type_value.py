from .. ndtypes import TypeValueT 
from expr import Expr 


class TypeValue(Expr):
  """
  Value materialization of a type 
  """
  def __init__(self, type_value, type = None, source_info = None):
    self.type_value = type_value
     
    if type is None: type = TypeValueT(self.type_value)
    assert isinstance(type, TypeValueT)
    assert type.type is not None 
    
    self.type = type 
     
    