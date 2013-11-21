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
  def __init__(self, fn, arg_types):
    
    self.fn = fn 
    
    if arg_types is None:
      arg_types = ()
    elif not hasattr(arg_types, '__iter__'):
      arg_types = tuple([arg_types])
    else:
      arg_types = tuple(arg_types)
      
    self.arg_types = arg_types 
    self._hash = hash( (fn,) + arg_types)
  
    self.specializations = {}

  def children(self):
    return self.arg_types
  
  def __hash__(self):
    return self._hash 

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
