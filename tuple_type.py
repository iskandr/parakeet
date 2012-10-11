from core_types import IncompatibleTypes, StructT

 
class TupleT(StructT):
  rank = 0 
  _members = ['elt_types']
  
  def finalize_init(self):
    self._fields_ = [ 
      ("elt%d" % i, t) for (i,t) in enumerate(self.elt_types)
    ]
                             
  def dtype(self):
    raise RuntimeError("Do tuples have dtypes?")
  
  def __eq__(self, other):
    return isinstance(other, TupleT) and self.elt_types == other.elt_types 

  def __hash__(self):
    return hash(self.elt_types)
  
  def combine(self, other):
    if isinstance(other, TupleT) and len(other.elt_types) == len(self.elt_types):
      combined_elt_types = [t1.combine(t2) for \
                            (t1, t2) in zip(self.elt_types, other.elt_tyepes)]
      if combined_elt_types != self.elt_types:
        return TupleT(combined_elt_types)
      else:
        return self
    else:
      raise IncompatibleTypes(self, other)  
  
_tuple_types = {}
def repeat_tuple(t, n):
  """Given the base type t, construct the n-tuple t*t*...*t"""
  elt_types = tuple([t] * n)
  if elt_types in _tuple_types:
    return _tuple_types[elt_types]
  else:
    tuple_t = TupleT(elt_types)
    _tuple_types[elt_types] = tuple_t
    return tuple_t 


def make_tuple_type(elt_types):
  """
  Use this memoized construct to avoid
  constructing too many distinct tuple type
  objects and speeding up equality checks
  """ 
  key = tuple(elt_types)
  if key in _tuple_types:
    return _tuple_types[key]
  else:
    t = TupleT(key)
    _tuple_types[key] = t
    return t
  

import type_conv 

def typeof(python_tuple):
  return make_tuple_type(map(type_conv.typeof, python_tuple))

def from_python(struct_repr, python_tuple):  
  converted_elts = [type_conv.from_python(elt) for elt in python_tuple]
  return struct_repr(*converted_elts)

def to_python(struct_obj, parakeet_type):
  python_elt_values = []
  for (field_name, elt_t) in parakeet_type._fields_:
    internal_field_value = getattr(struct_obj, field_name)
    python_elt_value = type_conv.to_python(internal_field_value, elt_t)
    python_elt_values.append(python_elt_value)
  return tuple(python_elt_values)

type_conv.register(tuple, TupleT, typeof, from_python, to_python)
