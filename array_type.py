
# import numpy as np 

from tuple_type import repeat_tuple
from core_types import StructT, IncompatibleTypes, ScalarT, Int64, ptr_type 

class ArrayT(StructT):
  _members = ['elt_type', 'rank']

  def finalize_init(self):
    assert isinstance(self.elt_type, ScalarT)
    
    self._field_types = [
      ('data', ptr_type(self.elt_type)), 
      ('shape', repeat_tuple(Int64, self.rank)), 
      ('strides', repeat_tuple(Int64, self.rank)),
    ]

  def dtype(self):
    return self.elt_type.dtype
 
  def __eq__(self, other): 
    return isinstance(other, ArrayT) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)

    
