import type_conv

from core_types import StructT, ImmutableT, IncompatibleTypes

class SliceT(StructT, ImmutableT):
  def __init__(self, start_type, stop_type, step_type):
    self.start_type = start_type
    self.stop_type = stop_type
    self.step_type = step_type
    self._fields_ = [('start',start_type), ('stop', stop_type), ('step', step_type)]
    self._hash = hash((self.start_type, self.stop_type, self.step_type))
  
  def children(self):
    yield self.start_type
    yield self.stop_type
    yield self.step_type
    
  def __eq__(self, other):
    return self is other or \
      (other.__class__ is SliceT and
       self.start_type == other.start_type and
       self.stop_type == other.stop_type and
       self.step_type == other.step_type)

  def __hash__(self):
    return self._hash

  def combine(self, other):
    if self == other: return self
    else:raise IncompatibleTypes(self, other)

  def __str__(self):
    return "SliceT(%s, %s, %s)" % (self.start_type,
                                   self.stop_type,
                                   self.step_type)

  def __repr__(self):
    return str(self)

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
