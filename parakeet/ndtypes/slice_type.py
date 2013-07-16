import type_conv

from core_types import StructT, ImmutableT, IncompatibleTypes

class SliceT(StructT, ImmutableT):
  _members = ['start_type', 'stop_type', 'step_type']

  def node_init(self):
    self._fields_ = [
      ('start', self.start_type),
      ('stop', self.stop_type),
      ('step', self.step_type),
    ]

  def __eq__(self, other):
    return self is other or \
      (other.__class__ is SliceT and
       self.start_type == other.start_type and
       self.stop_type == other.stop_type and
       self.step_type == other.step_type)

  def __hash__(self):
    return hash((self.start_type, self.stop_type, self.step_type))

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
