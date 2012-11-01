from core_types import Type, IncompatibleTypes, StructT, Int64
import type_conv 

# python's function type 
from types import FunctionType
import ast_conversion 
import closure_signatures 


###########################################
#
#  Closures! 
#
###########################################



class ClosureT(StructT):
  _members = ['fn', 'args']
  
  def node_init(self):
    if self.args is None:
      self.args = ()
    elif not hasattr(self.args, '__iter__'):
      self.args = tuple([self.args])
    elif not isinstance(self.args, tuple):
      self.args = tuple(self.args)
    
    print "CLOSURE CREATE", self.fn, self.args
    
    self._fields_ = [('fn_id', Int64)] 
    for (i, t) in enumerate(self.args):
      self._fields_.append( ('arg%d' % i, t) )
    
  def __hash__(self):
    return hash(repr(self))
  
  def __eq__(self, other):
    return self.fn == other.fn and self.args == other.args
  
  def from_python(self, python_fn):
    untyped_fundef = ast_conversion.translate_function_value(python_fn)
    closure_args = untyped_fundef.python_nonlocals()
    closure_arg_types = map(type_conv.typeof, closure_args)
    
    closure_t = make_closure_type(untyped_fundef, closure_arg_types)
    closure_id = closure_signatures.get_id(closure_t)
    converted_args = map(type_conv.from_python, closure_args)
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
def make_closure_type(untyped_fn, closure_arg_types = []):
  name = untyped_fn.name
  closure_arg_types = tuple(closure_arg_types)
  key = (name, closure_arg_types)
  if key in _closure_type_cache:
    return _closure_type_cache[key]
  else:
    t = ClosureT(name, closure_arg_types)
    _closure_type_cache[key] = t 
    return t
  

def typeof_fn(f):
  untyped_fn = ast_conversion.translate_function_value(f)
  closure_args = untyped_fn.python_nonlocals()
  closure_arg_types = map(type_conv.typeof, closure_args)
  return make_closure_type(untyped_fn, closure_arg_types)


type_conv.register(FunctionType, ClosureT, typeof_fn)

import prims

def typeof_prim(p):
  untyped_fn = prims.prim_wrapper(p)
  return make_closure_type(untyped_fn, [])

type_conv.register(prims.class_list, ClosureT, typeof_prim)
  
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