from core_types import ConcreteT
from scalar_types import IntT 

###########################################
#
#  Pointers!
#
###########################################

class PtrT(ConcreteT):
  """
  I'm not giving pointer a concrete to_python/from_python conversion or any
  usable fields, so it's up to our ctypes_repr and llvm_backend to appropriately
  interpret objects of this type.
  """
  def __init__(self, elt_type):
    self.elt_type = elt_type 
    self._hash =  hash(elt_type)
    
  rank = 1
  def index_type(self, idx):
    assert isinstance(idx, IntT), \
        "Index into pointer must be of type int, got %s" % (idx)
    return self.elt_type

  def children(self):
    return self.elt_type

  def __str__(self):
    return "ptr(%s)" % self.elt_type

  def __eq__(self, other):
    return other.__class__ is PtrT and self.elt_type == other.elt_type

  def __hash__(self):
    return self._hash 

  def __repr__(self):
    return str(self)

  @property
  def ctypes_repr(self):
    return self._ctypes_repr

_ptr_types = {}
def ptr_type(t):
  if t in _ptr_types:
    return _ptr_types[t]
  else:
    ptr_t = PtrT(t)
    _ptr_types[t] = ptr_t
    return ptr_t
