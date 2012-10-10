import numpy as np 
import ctypes 
import abc 
from node import Node





class StructT(ConcreteT):
  """
  Structs must define how to translate themselves
  from their normal python representation to a simplified 
  form involving only base types. 
   
  Any derived class *must* define a _field_types property
  which contains a list of (name, field_type) pairs.
  """
  pass


class TupleT(StructT):
  rank = 0 
  _members = ['elt_types']
  
  def finalize_init(self):
    self._field_types = [ 
      ("elt%d" % i, t) for (i,t) in enumerate(self.elt_types)
    ]
    self.ctypes_repr = ctypes_struct_from_fields(self._field_types)                           

  def to_ctypes(self, python_tuple):
    assert isinstance(python_tuple, tuple)
    assert len(python_tuple) == len(self.elt_types)
    
    converted_elts = []
    for (elt_type, elt_value) in zip(self.elt_types, python_tuple):
      converted_elts.append( elt_type.to_ctypes(elt_value) )
      
    assert isinstance(self.ctypes_repr, ctypes.Structure)
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


class ArrayT(StructT):
  _members = ['elt_type', 'rank']

  def finalize_init(self):
    assert isinstance(self.elt_type, ScalarT)
    tuple_t = repeat_tuple(Int64, self.rank)
    
    self._field_types = [
      ('data', make_buffer_type(self.elt_type)), 
      ('shape', tuple_t), 
      ('strides', tuple_t),
    ]
    self.ctypes_repr = ctypes_struct_from_fields(self._field_types)  

  def to_ctypes(self, x):
    return self.ctypes_repr()
  
  def from_ctypes(self, x):
    """
    WARNING! 
    If a pointer gets passed into and back out of 
    parakeet we're going to screw up the reference counts
    on its buffer! 
    """
    assert False, "Haven't yet implemented construction of buffers"
    
  def nbytes(self):
    raise RuntimeError("Can't get size of an array just from its type")  

  def dtype(self):
    return self.elt_type.dtype()
 
  def __eq__(self, other): 
    return isinstance(other, ArrayT) and \
      self.elt_type == other.elt_type and self.rank == other.rank

  def combine(self, other):
    if self == other:
      return self
    else:
      raise IncompatibleTypes(self, other)

    

class ClosureT(StructT):
  _members = ['fn', 'args']
  
  def finalize_init(self):
    if self.args is None:
      self.args = ()
    elif not hasattr(self.args, '__iter__'):
      self.args = tuple([self.args])
    elif not isinstance(self.args, tuple):
      self.args = tuple(self.args)
      
    self._fields_types = [('fn_id', Int64)] 
    for (i, t) in enumerate(self.args):
      self._field_types.append( ('arg%d' % i, t.ctypes_repr) )
      
    self.ctypes_repr = ctypes_struct_from_fields(self._fields_types) 

  def to_ctypes(self, fn_val):
    assert False, "Not yet sure how to translate closure types from python"
  
  def from_ctypes(self, clos):
    assert False, "Not yet sure how to return closures back to python"
    
    
  def __hash__(self):
    return hash(repr(self))
  
  def __eq__(self, other):
    return self.fn == other.fn and self.args == other.args
  
  def combine(self, other):
    if isinstance(other, ClosureSet):
      return other.combine(self)
    elif isinstance(other, ClosureT):
      if self == other:
        return self
      else:
        return ClosureSet(self, other)
    else:
      raise IncompatibleTypes(self, other)
    
class ClosureSet(Type):
  """
  If multiple closures meet along control flow paths then join them into a closure set.
  This type should not appear by the time we're generating LLVM code.
  """
  _members = ['closures'] 
  
  def __init__(self, *closures):
    self.closures = set([])
    for clos_t in closures:
      if isinstance(clos_t, ClosureSet):
        self.closures.update(clos_t.closures)
      else:
        assert isinstance(clos_t, ClosureT)
        self.closures.add(clos_t)

  def combine(self, other):
    if isinstance(other, ClosureSet):
      combined_closures = self.closures.union(other.closures)
      if combined_closures != self.closures:
        return ClosureSet(combined_closures)
      else:
        return self 
    else:
      raise IncompatibleTypes(self, other)
  
  def __eq__(self, other):
    return self.closures == other.closures 
  
  def __iter__(self):
    return iter(self.closures)
  
  def __len__(self):
    return len(self.closures)

  
def type_of_scalar(x):
  if isinstance(x, bool):
    return Bool 
  elif isinstance(x, (int, long)):
    return Int64
  elif isinstance(x, float):
    return Float64
  else:
    raise RuntimeError("Unsupported scalar type: %s" % x)
  
def type_of_value(x):
  if np.isscalar(x):
    return type_of_scalar(x)
  elif isinstance(x, tuple):
    elt_types = map(type_of_value, x)
    return TupleT(elt_types)
  elif isinstance(x, np.ndarray):
    return ArrayT(from_dtype(x.dtype), np.rank(x))
  else:
    raise RuntimeError("Unsupported type " + str(type(x)))
  
def combine_type_list(types):
  common_type = Unknown 

  for t in types:
    common_type = common_type.combine(t)
  return common_type

