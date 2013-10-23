
import ctypes
import numpy as np
import type_conv

import core_types

from core_types import StructT, IncompatibleTypes, NoneT 
from ptr_type import ptr_type
from scalar_types import Int64, ScalarT, IntT 
from tuple_type import TupleT, repeat_tuple
from slice_type import SliceT, make_slice_type  
def buffer_info(buf, ptr_type = ctypes.c_void_p):
  """Given a python buffer, return its address and length"""

  assert isinstance(buf, buffer)
  address = ptr_type()
  length = ctypes.c_ssize_t()
  obj = ctypes.py_object(buf)
  ctypes.pythonapi.PyObject_AsReadBuffer(obj, ctypes.byref(address),
                                         ctypes.byref(length))
  return address, length.value

ctypes.pythonapi.PyBuffer_New.argtypes = (ctypes.c_ulong,)
ctypes.pythonapi.PyBuffer_New.restype = ctypes.py_object

PyBuffer_New = ctypes.pythonapi.PyBuffer_New
PyBuffer_FromReadWriteMemory = ctypes.pythonapi.PyBuffer_FromReadWriteMemory

class ArrayT(StructT):
  _members = ['elt_type', 'rank']

  def node_init(self):
    assert isinstance(self.elt_type, ScalarT), \
      "Can't create array with element type %s, currently only scalar elements supported" % \
      (self.elt_type,)
        
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
  
  def dtype(self):
    return self.elt_type.dtype

  def __str__(self):
    return "array%d(%s)" % (self.rank, self.elt_type)

  def __eq__(self, other):
    return other.__class__ is ArrayT and \
        self.elt_type == other.elt_type and \
        self.rank == other.rank

  def __hash__(self):
    return hash((self.elt_type, self.rank))

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

    if not isinstance(idx, core_types.Type):
      idx = idx.type

    if idx.__class__ is TupleT:
      indices = idx.elt_types
    else:
      indices = [idx]

    n_indices = len(indices)
    n_required = self.rank
    if n_required > n_indices:
      n_missing = n_required - n_indices
      extra_indices = [core_types.NoneType] * n_missing
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

  # WARNING:
  # until we have garbage collection figured out, we'll
  # leak memory from arrays we allocate in the conversion routine
  _store_forever = []
  _seen_ptr = set([])
  def from_python(self, x):
    if not isinstance(x, np.ndarray):
      print "Warning: Copying %s into Parakeet, will never get deleted" % \
          (type(x))
      x = np.asarray(x)
      self._store_forever.append(x)

    
    nelts = reduce(lambda x,y: x*y, x.shape)
    elt_size = x.dtype.itemsize
    # total_bytes = nelts * elt_size
    if x.base is not None:
      if isinstance(x.base, np.ndarray):
        ptr = x.base.ctypes.data_as(self.ptr_t.ctypes_repr)
        offset_bytes = x.ctypes.data  - x.base.ctypes.data 
      else:
        assert isinstance(x.base, buffer)
        ptr, _ = buffer_info(x.base, self.ptr_t.ctypes_repr)
        offset_bytes = x.ctypes.data - ctypes.addressof(ptr.contents) 
      offset = offset_bytes / elt_size
    else:
      ptr = x.ctypes.data_as(self.ptr_t.ctypes_repr)
      offset = 0
   
    ctypes_shape = self.shape_t.from_python(x.shape)
    strides_in_elts = tuple([s / elt_size for s in x.strides])
    ctypes_strides = self.strides_t.from_python(strides_in_elts)
    return self.ctypes_repr(ptr, 
                            ctypes.pointer(ctypes_shape),
                            ctypes.pointer(ctypes_strides), 
                            offset, 
                            nelts)
    
  def to_python(self, obj):
    """
    For now, to avoid to dealing with the messiness of ownership, we just always
    copy data on the way out of Parakeet
    """

    shape = self.shape_t.to_python(obj.shape.contents)

    elt_size = self.elt_type.nbytes
    strides_in_elts = self.strides_t.to_python(obj.strides.contents)
    strides_in_bytes = tuple([s * elt_size for s in strides_in_elts])

    base_ptr = obj.data

    nbytes = obj.size * elt_size

    dest_buf = PyBuffer_New(nbytes)
    dest_ptr, _ = buffer_info(dest_buf, self.ptr_t.ctypes_repr)
    ctypes.memmove(dest_ptr, base_ptr, nbytes)

    
    return np.ndarray(shape, dtype = self.elt_type.dtype,
                      buffer = dest_buf,
                      strides = strides_in_bytes,
                      offset = obj.offset * elt_size)

_array_types = {}
def make_array_type(elt_t, rank):
  key = (elt_t, rank)
  if key in _array_types:
    return _array_types[key]
  else:
    if rank == 0:
      t = elt_t
    else:
      t = ArrayT(elt_t, rank)
    _array_types[key] = t
    return t


def elt_type(t):
  if t.__class__ is ArrayT:
    return t.elt_type
  else:
    return t

def elt_types(ts):
  return map(elt_type, ts)

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
