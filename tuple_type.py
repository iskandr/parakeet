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
  

def to_ctypes(self, python_tuple):
  assert isinstance(python_tuple, tuple)
  
  
  assert len(python_tuple) == len(self.elt_types)
    
  converted_elts = []
  for (elt_type, elt_value) in zip(self.elt_types, python_tuple):
    converted_elts.append( elt_type.to_ctypes(elt_value) )
      
  ctypes_struct_t = ctypes_repr.to_ctypes()
  assert len(self.ctypes_repr._fields_) == len(converted_elts)
  return self.ctypes_repr(*converted_elts)
  
def from_ctypes(self, struct):
  python_elt_values = []
  for (i, elt_t) in enumerate(self.elt_types):
    field_name = "elt%d" % i
    internal_field_value = getattr(struct, field_name)
    python_elt_value = elt_t.from_ctypes(internal_field_value)
    python_elt_values.append(python_elt_value)
  return tuple(python_elt_values)

 