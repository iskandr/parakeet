import core_types
import ctypes
import numpy as np
import type_conv

from core_types import StructT, IncompatibleTypes, ScalarT, Int64, ptr_type
from tuple_type import TupleT, repeat_tuple

def buffer_info(buf, ptr_type = ctypes.c_void_p):
  """Given a python buffer, return its address and length"""
  assert isinstance(buf, buffer)
  address = ptr_type()
  length = ctypes.c_ssize_t()
  obj =  ctypes.py_object(buf)
  ctypes.pythonapi.PyObject_AsReadBuffer(obj, ctypes.byref(address),
                                         ctypes.byref(length))
  return address, length.value

ctypes.pythonapi.PyBuffer_New.argtypes = (ctypes.c_ulong,)
ctypes.pythonapi.PyBuffer_New.restype = ctypes.py_object
AllocateBuffer = ctypes.pythonapi.PyBuffer_New

class SliceT(StructT):
  _members = ['start_type', 'stop_type', 'step_type']

  def node_init(self):
    self._fields_ = [
      ('start', self.start_type),
      ('stop', self.stop_type),
      ('step', self.step_type),
    ]

  def __eq__(self, other):
    return self is other or \
      (self.start_type == other.start_type and
       self.stop_type == other.stop_type and
       self.step_type == other.step_type)

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)

  def __str__(self):
    return "SliceT(%s, %s, %s)" % (self.start_type, self.stop_type,
                                   self.step_type)

  def __repr__(self):
    return str(self)

  def from_python(self, py_slice):
    start = self.start_type.from_python(py_slice.start)
    stop = self.stop_type.from_python(py_slice.stop)
    step = self.step_type.from_python(py_slice.step)
    return self.ctypes_repr(start, stop, step)

  def to_python(self, obj):
    start = self.start_type.to_python(obj.start)
    stop = self.stop_type.to_python(obj.stop)
    step = self.step_type.to_python(obj.step)
    return slice(start, stop, step)

_slice_type_cache = {}
def make_slice_type(start_t, stop_t, step_t):
  key = (start_t, stop_t, step_t)
  if key in _slice_type_cache:
    return _slice_type_cache[key]
  else:
    t = SliceT(start_t, stop_t, step_t)
    _slice_type_cache[key] = t
    return t

def typeof_slice(s):
  start_type = type_conv.typeof(s.start)
  stop_type = type_conv.typeof(s.stop)
  step_type = type_conv.typeof(s.step)
  return make_slice_type(start_type, stop_type, step_type)

type_conv.register(slice, SliceT, typeof_slice)

class ArrayT(StructT):
  _members = ['elt_type', 'rank']

  def node_init(self):
    tuple_t = repeat_tuple(Int64, self.rank)

    self.shape_t = tuple_t
    self.strides_t = tuple_t
    self.ptr_t = ptr_type(self.elt_type)
    self._fields_ = [
      ('data', self.ptr_t),
      ('shape', tuple_t),
      ('strides', tuple_t),
    ]

  def dtype(self):
    return self.elt_type.dtype

  def __str__(self):
    return "array%d(%s)" % (self.rank, self.elt_type)

  def __eq__(self, other):
    return isinstance(other, ArrayT) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    elif isinstance(other, core_types.ScalarT):
      elt_t = self.elt_type
      combined_elt_t = elt_t.combine(other)
      return make_array_type(combined_elt_t, self.rank)
    elif isinstance(other, ArrayT):
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

    if isinstance(idx, TupleT):
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
      if isinstance(t, core_types.IntT):
        result_rank -= 1
      else:
        assert isinstance(t, (core_types.NoneT, SliceT, ArrayT, TupleT)), \
            "Unexpected index type: %s " % t
    if result_rank > 0:
      return make_array_type(self.elt_type, result_rank)
    else:
      return self.elt_type

  # WARNING:
  # until we have garbage collection figured out, we'll
  # leak memory from arrays we allocate in the conversion routine
  _store_forever = []
  def from_python(self, x):

    if not isinstance(x, np.ndarray):
      x = np.asarray(x)
      self._store_forever.append(x)

    ptr, buffer_length = buffer_info(x.data, self.ptr_t.ctypes_repr)

    nelts = reduce(lambda x,y: x*y, x.shape)
    elt_size = x.dtype.itemsize
    total_bytes = nelts * elt_size
    assert total_bytes == buffer_length, \
        "Shape %s has %d elements of size %d (total = %d) but buffer has" + \
        " length %d bytes" % \
        (x.shape, nelts, elt_size, total_bytes, buffer_length)
    ctypes_shape = self.shape_t.from_python(x.shape)

    strides_in_elts = tuple([s / elt_size for s in x.strides])
    ctypes_strides = self.strides_t.from_python(strides_in_elts)
    return self.ctypes_repr(ptr, ctypes.pointer(ctypes_shape),
                            ctypes.pointer(ctypes_strides))

  def to_python(self, obj):
    """
    For now, to avoid to dealing with the messiness of ownership,
    we just always copy data on the way out of Parakeet
    """
    shape = self.shape_t.to_python(obj.shape.contents)

    elt_size = self.elt_type.nbytes
    strides_in_elts = self.strides_t.to_python(obj.strides.contents)
    assert any([stride == 1 for stride in strides_in_elts]), \
        "Discontiguous array not supported, strides = %s" % (strides_in_elts,)
    strides_in_bytes = tuple([s * elt_size for s in strides_in_elts])

    n_elts = np.prod(shape)
    n_bytes = n_elts * elt_size
    dest_buf = AllocateBuffer(n_bytes)
    dest_ptr, _ = buffer_info(dest_buf, self.ptr_t.ctypes_repr)

    # copy data from pointer
    ctypes.memmove(dest_ptr, obj.data, n_bytes)
    return np.ndarray(shape, dtype = self.elt_type.dtype,
                      buffer = dest_buf, strides = strides_in_bytes)

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

def typeof_array(x):
  x = np.asarray(x)
  elt_t = core_types.from_dtype(x.dtype)
  rank = len(x.shape)
  return make_array_type(elt_t, rank)

type_conv.register((np.ndarray, list),  ArrayT, typeof_array)

def elt_type(t):
  if isinstance(t, ArrayT):
    return t.elt_type
  else:
    return t

def elt_types(ts):
  return map(elt_type, ts)

def lower_rank(t, r):
  if isinstance(t, ArrayT):
    assert t.rank >= r
    return make_array_type(t.elt_type, t.rank - r)
  else:
    return t

def increase_rank(t, r):
  if isinstance(t, ArrayT):
    return make_array_type(t.elt_type, t.rank + r)
  else:
    return make_array_type(t, r)

def lower_ranks(arg_types, r):
  return [lower_rank(t, r) for t in arg_types]
