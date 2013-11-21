from core_types import StructT, IncompatibleTypes, NoneT, Type, NoneType   
from ptr_type import ptr_type
from scalar_types import Int64, ScalarT, IntT 
from tuple_type import TupleT, repeat_tuple
from slice_type import SliceT
  
class ArrayT(StructT):
  
  def __init__(self, elt_type, rank):
    assert isinstance(elt_type, ScalarT), \
      "Can't create array with element type %s, currently only scalar elements supported" % \
      (self.elt_type,)
    self.elt_type = elt_type
    self.rank = rank 
    
    tuple_t = repeat_tuple(Int64, self.rank)

    self.shape_t = tuple_t
    self.strides_t = tuple_t
    self.ptr_t = ptr_type(self.elt_type)
    self._fields_ = [
      ('data', self.ptr_t),
      ('shape', tuple_t),
      ('strides', tuple_t),
      ('offset', Int64),
      ('size', Int64),
      # ('dtype', TypeValueT(self.elt_type))
    ]
    self._hash = hash( (elt_type, rank) )
  
  def children(self):
    yield self.elt_type
    yield self.shape_t
    yield self.strides_t
    yield self.ptr_t 
     
  def dtype(self):
    return self.elt_type.dtype

  def __str__(self):
    return "array%d(%s)" % (self.rank, self.elt_type)

  def __eq__(self, other):
    if self is other:
      return True 
    return other.__class__ is ArrayT and \
        self.elt_type == other.elt_type and \
        self.rank == other.rank

  def __hash__(self):
    return self._hash 

  def combine(self, other):
    if self == other:
      return self
    elif isinstance(other, ScalarT):
      elt_t = self.elt_type
      combined_elt_t = elt_t.combine(other)
      return make_array_type(combined_elt_t, self.rank)
    elif other.__class__ is ArrayT:
      assert self.rank == other.rank
      combined_elt_t = self.elt_type.combine(other.elt_type)
      return make_array_type(combined_elt_t, self.rank)
    else:
      raise IncompatibleTypes(self, other)

  def index_type(self, idx):
    """
    Given the type of my indices, what type of array will result?
    """

    if not isinstance(idx, Type):
      idx = idx.type

    if idx.__class__ is TupleT:
      indices = idx.elt_types
    else:
      indices = [idx]

    n_indices = len(indices)
    n_required = self.rank
    if n_required > n_indices:
      n_missing = n_required - n_indices
      extra_indices = [NoneType] * n_missing
      indices = indices + extra_indices

    # we lose one result dimension for each int in the index set
    result_rank = n_required
    for t in indices:
      if isinstance(t, IntT):
        result_rank -= 1
      else:
        assert isinstance(t,  (TupleT, NoneT, SliceT, ArrayT, )),  "Unexpected index type: %s " % t
    if result_rank > 0:
      return make_array_type(self.elt_type, result_rank)
    else:
      return self.elt_type

_array_types = {}
def make_array_type(elt_t, rank):
  key = (elt_t, rank)
  arr_t = _array_types.get(key)
  if arr_t is None: 
    if rank == 0:  
      arr_t = elt_t 
    else: 
      arr_t = ArrayT(elt_t, rank)
    _array_types[key] = arr_t
  return arr_t

def elt_type(t):
  if t.__class__ is ArrayT:
    return t.elt_type
  else:
    return t

def elt_types(ts):
  return [elt_type(t) for t in ts]

def lower_rank(t, r):
  if t.__class__ is ArrayT:
    assert t.rank >= r
    return make_array_type(t.elt_type, t.rank - r)
  else:
    return t

def increase_rank(t, r):
  if t.__class__ is ArrayT:
    return make_array_type(t.elt_type, t.rank + r)
  else:
    return make_array_type(t, r)

def lower_ranks(arg_types, r):
  return [lower_rank(t, r) for t in arg_types]

def get_rank(t):
  if t.__class__ is ArrayT:
    return t.rank
  else:
    return 0

def rank(t):
  return get_rank(t)
