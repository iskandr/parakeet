import numpy as np 


from .. syntax.helpers import const 
from ..transforms  import Transform, Simplify, Phase, DCE 


from find_constant_values import symbolic_call

from abstract_value import ( 
   Const, Array, Struct, Tuple, 
   specialization_const, abstract_tuple, abstract_array, 
   unknown, zero, one 
)

class ValueSpecializer(Transform):
  def __init__(self, abstract_inputs):
    Transform.__init__(self)
    self.abstract_inputs = abstract_inputs 
  
  def pre_apply(self, fn):
    env, _ = symbolic_call(fn, self.abstract_inputs)
    self.env = env 
  
  def transform_Var(self, expr):
    if expr.name in self.env:
      abstract_value = self.env[expr.name]
      if abstract_value.__class__ is Const:
        expr.value = expr.type.convert_python_value(abstract_value.value)
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


def from_python(python_value):
  if isinstance(python_value, np.ndarray):
    elt_size = python_value.dtype.itemsize 
    strides = []
    for s in python_value.strides:
      strides.append(specialization_const(s/elt_size)) 
    return abstract_array(strides)
  elif isinstance(python_value, tuple):
    return abstract_tuple(from_python_list(python_value))
  elif python_value == 0:
    return zero 
  elif python_value == 1:
    return one 
  else:
    return unknown 
    
  
def from_python_list(python_values):
  return tuple([from_python(v) for v in python_values]) 

_cache = {}
def specialize(fn, python_values):
  abstract_values = from_python_list(python_values)
  key = (fn.cache_key, abstract_values)
  if key in _cache:
    return _cache[key]

  if any(has_unit_stride(v) for v in abstract_values):
    specializer = ValueSpecializer(abstract_values)
    transforms = Phase([specializer, Simplify, DCE],
                        memoize = False, 
                        copy = True, 
                        name = "StrideSpecialization for %s" % (abstract_values,), 
                        recursive = False)
    new_fn = transforms.apply(fn)
  else:
    new_fn = fn
  _cache[key] = new_fn
  return new_fn