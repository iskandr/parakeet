import ctypes
from core_types import Type, IncompatibleTypes, StructT, ImmutableT
from scalar_types import  Int64
import type_conv 

###########################################
#
#  Closures!
#
###########################################

class ClosureT(StructT, ImmutableT):
  _members = ['fn', 'arg_types']

  max_id = 0
  # map each distinct closure_type to an integer id
  id_numbers = {}

  def node_init(self):
    if self.arg_types is None:
      self.arg_types = ()
    elif not hasattr(self.arg_types, '__iter__'):
      self.arg_types = tuple([self.arg_types])
    elif not isinstance(self.arg_types, tuple):
      self.arg_types = tuple(self.arg_types)

    self._fields_ = [] #('fn_id', Int64)]
    for (i, t) in enumerate(self.arg_types):
      self._fields_.append( ('arg%d' % i, t) )

    self.specializations = {}
    if self in self.id_numbers:
      self.id = self.id_numbers[self]
    else:
      self.id = self.max_id
      self.id_numbers[self] = self.id
      self.max_id += 1

  def __hash__(self):
    return hash( (self.fn,) + self.arg_types)

  def __eq__(self, other):
    
    if other.__class__ is not ClosureT:
      return False
    if self.fn.__class__ != other.fn.__class__:
      return False
    if isinstance(self.fn, str):
      same_fn = (self.fn == other.fn)
    else:
      # checking equality of functions isn't really defined, 
      # so just check that it's the same location 
      same_fn = self.fn is other.fn
    return same_fn and self.arg_types == other.arg_types

  def __str__(self):
    fn_name = self.fn if isinstance(self.fn, str) else self.fn.name
    return "ClosT(%s, {%s})" % (fn_name, ", ".join(str(t)
                                                   for t in self.arg_types))


  
  def from_python(self, python_fn):
    from ..frontend import ast_conversion
    
    untyped_fundef = ast_conversion.translate_function_value(python_fn)
    closure_args = untyped_fundef.python_nonlocals()
    closure_arg_types = map(type_conv.typeof, closure_args)

    closure_t = make_closure_type(untyped_fundef, closure_arg_types)
    closure_id = id_of_closure_type(closure_t)

    def field_value(closure_arg):
      obj = type_conv.from_python(closure_arg)
      parakeet_type = type_conv.typeof(closure_arg)
      if isinstance(parakeet_type, StructT):
        return ctypes.pointer(obj)
      else:
        return obj

    converted_args = [field_value(closure_arg) for closure_arg in closure_args]
    return closure_t.ctypes_repr(closure_id, *converted_args)


  def to_python(self, parakeet_fn):
    raise RuntimeError("TODO: Just return a compiled_fn wrapper")

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

_closure_type_cache = {}
def make_closure_type(fn, closure_arg_types = []):
  closure_arg_types = tuple(closure_arg_types)
  key = (fn, closure_arg_types)
  if key in _closure_type_cache:
    return _closure_type_cache[key]
  else:
    t = ClosureT(fn, closure_arg_types)
    _closure_type_cache[key] = t
    return t




"""
Map each (untyped fn id, fixed arg) types to a distinct integer so that the
runtime representation of closures just need to carry this ID
"""

closure_type_to_id = {}
id_to_closure_type = {}
max_id = 0

def id_of_closure_type(closure_t):
  global max_id
  assert isinstance(closure_t, ClosureT), \
      "Expected closure type, got: " + str(closure_t)
  if closure_t in closure_type_to_id:
    return closure_type_to_id[closure_t]
  else:
    num = max_id
    max_id += 1
    closure_type_to_id[closure_t] = num
    return num

def closure_type_from_id(num):
  assert num in id_to_closure_type
  return id_to_closure_type[num]



class ClosureSet(Type):
  """
  If multiple closures meet along control flow paths then join them into a
  closure set. This type should not appear by the time we're generating LLVM
  code.
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
