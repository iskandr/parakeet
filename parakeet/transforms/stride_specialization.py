from .. analysis.find_constant_strides import FindConstantStrides, Const, Array, Struct, Tuple
from .. analysis.find_constant_strides import from_python_list, from_internal_repr
from .. syntax.helpers import const_int, const 
from dead_code_elim import DCE
from phase import Phase
from simplify import Simplify
from transform import Transform

class StrideSpecializer(Transform):
  def __init__(self, abstract_inputs):
    Transform.__init__(self)
    self.abstract_inputs = abstract_inputs 
  
  def pre_apply(self, fn):
    analysis = FindConstantStrides(fn, self.abstract_inputs)
    analysis.visit_fn(fn)
    self.env = analysis.env

  
  def transform_Var(self, expr):
    if expr.name in self.env:
      value = self.env[expr.name]
      if value.__class__ is Const:
        result = const(value.value)
        return result 
    return expr
  
  def transform_lhs(self, lhs):
    return lhs
  
def has_unit_stride(abstract_value):
  c = abstract_value.__class__
  if c is Array:
    return has_unit_stride(abstract_value.strides)
  elif c is Struct:
    return any(has_unit_stride(field_val) 
               for field_val 
               in abstract_value.fields.itervalues())
  elif c is Tuple:
    return any(has_unit_stride(elt) 
               for elt in abstract_value.elts)
  elif c is Const:
    return abstract_value.value == 1
  else:
    return False
  
_cache = {}
def specialize(fn, python_values, types = None):
  if types is None:
    abstract_values = from_python_list(python_values)
  else:
    # if types are given, assume that the values 
    # are already converted to Parakeet's internal runtime 
    # representation 
    abstract_values = []
    for (t, internal_value) in zip(types, python_values):
      abstract_values.append(from_internal_repr(t, internal_value))
  
  key = (fn.cache_key, tuple(abstract_values))
  if key in _cache:
    return _cache[key]
  elif any(has_unit_stride(v) for v in abstract_values):
    specializer = StrideSpecializer(abstract_values)

    transforms = Phase([specializer, Simplify, DCE],
                        memoize = False, copy = True, 
                        name = "StrideSpecialization for %s" % abstract_values, 
                        recursive = False)
    new_fn = transforms.apply(fn)

  else:
    new_fn = fn
  _cache[key] = new_fn
  return new_fn