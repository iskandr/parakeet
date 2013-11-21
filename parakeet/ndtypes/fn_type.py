from core_types import  IncompatibleTypes, ImmutableT

class FnT(ImmutableT):
  """Type of a typed function"""
  def __init__(self, input_types, return_type):
    self.input_types = tuple(input_types)
    self.return_type = return_type 
    self._hash = hash(self.input_types + (return_type,))

  def __str__(self):
    input_str = ", ".join(str(t) for t in self.input_types)
    return "(%s)->%s" % (input_str, self.return_type)

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return other.__class__ is FnT and \
           self.return_type == other.return_type and \
           len(self.input_types) == len(other.input_types) and \
           all(t1 == t2 for (t1, t2) in
               zip(self.input_types, other.input_types))

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)

  def __hash__(self):
    return self._hash

_fn_type_cache = {}
def make_fn_type(input_types, return_type):
  input_types = tuple(input_types)
  key = input_types, return_type
  if key in _fn_type_cache:
    return _fn_type_cache[key]
  else:
    t = FnT(input_types, return_type)
    _fn_type_cache[key] = t
    return t