import numpy as np 

from .. analysis.find_constant_strides import (
   FindConstantStrides, 
   Const, Array, Struct, Tuple, 
   unknown, specialization_const, abstract_tuple, abstract_array
)

from ..ndtypes import type_conv, TupleT, ArrayT, StructT
from .. syntax.helpers import const 
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



def from_internal_repr(parakeet_type, v):
  if v is None:
    return unknown
  elif hasattr(v, 'contents'):
    v = v.contents
    
  if parakeet_type.__class__ is TupleT:
    elts = []
    for (i,elt_t) in enumerate(parakeet_type.elt_types):
      elt_value = hasattr(v, "elt%d" % i)
      elts.append(from_internal_repr(elt_t, elt_value))
    return abstract_tuple(elts)
  elif parakeet_type.__class__ is ArrayT:
    strides_field = getattr(v, 'strides').contents
    strides = []
    for i in xrange(parakeet_type.rank):
      s = int(getattr(strides_field, 'elt%d'%i))
      strides.append(specialization_const(s))
    return abstract_array(strides)  
  elif isinstance(parakeet_type, StructT):
    fields = {}
    for (field_name, field_type) in parakeet_type._fields_:
      field_value = getattr(v, field_name)
      abstract_field = from_internal_repr(field_type, field_value)
      fields[field_name] = abstract_field
    return Struct(fields)        
  return unknown

def from_python(python_value):
  if isinstance(python_value, np.ndarray):
    elt_size = python_value.dtype.itemsize 
    strides = []
    for s in python_value.strides:
      strides.append(specialization_const(s/elt_size)) 
    return abstract_array(strides)
  elif isinstance(python_value, tuple):
    return abstract_tuple(from_python_list(python_value))
  else:
    parakeet_type = type_conv.typeof(python_value)
    parakeet_value = type_conv.from_python(python_value)
    return from_internal_repr(parakeet_type, parakeet_value)
  
def from_python_list(python_values):
  return [from_python(v) for v in python_values] 

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
  
  abstract_values = tuple(abstract_values)
  key = (fn.cache_key, abstract_values)
  result = _cache.get(key)
  if result is not None:
    return result 
  elif any(has_unit_stride(v) for v in abstract_values):
    specializer = StrideSpecializer(abstract_values)

    transforms = Phase([specializer, Simplify, DCE],
                        memoize = False, copy = True, 
                        name = "StrideSpecialization for %s" % (abstract_values,), 
                        recursive = False)
    new_fn = transforms.apply(fn)
  else:
    new_fn = fn
  _cache[key] = new_fn
  return new_fn