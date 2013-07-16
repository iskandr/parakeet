import ctypes 

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

  _members = ['elt_type']

  rank = 1
  def index_type(self, idx):
    assert isinstance(idx, IntT), \
        "Index into pointer must be of type int, got %s" % (idx)
    return self.elt_type

  def node_init(self):
    self._ctypes_repr = ctypes.POINTER(self.elt_type.ctypes_repr)

  def __str__(self):
    return "ptr(%s)" % self.elt_type

  def __eq__(self, other):
    return isinstance(other, PtrT) and self.elt_type == other.elt_type

  def __hash__(self):
    return hash(self.elt_type)

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
