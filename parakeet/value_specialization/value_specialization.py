import numpy as np 
from numpy import ndarray 

from .. import syntax 
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
  
  def lookup_expr(self, expr):
    c = expr.__class__ 
    if c is syntax.Var:
      return self.env.get(expr.name, unknown)
    else:
      return unknown 
    
  def lookup_expr_list(self, exprs):
    return [self.lookup_expr(e) for e in exprs]
  
  def transform_ParFor(self, stmt):
    fn = self.get_fn(stmt.fn)
    closure_elts = self.closure_elts(stmt.fn)
    abstract_values = self.lookup_expr_list(closure_elts)
    # all the index arguments have to be marked unknown 
    for _ in xrange(len(abstract_values), len(fn.input_types)):
      abstract_values.append(unknown)
    new_fn = specialize_abstract_values(fn, tuple(abstract_values))
    stmt.fn = self.closure(new_fn, closure_elts)
    return stmt 
    
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
  
  if c is Const:
    return abstract_value.value == 1
  elif c is Array:
    return has_unit_stride(abstract_value.strides)
  elif c is Tuple:
    return any(has_unit_stride(elt) 
               for elt in abstract_value.elts)
  elif c is Struct:
    return any(has_unit_stride(field_val) 
               for field_val 
               in abstract_value.fields.itervalues())
  else:
    return False


def from_python(python_value):
  t = type(python_value)
  if t is ndarray:
    elt_size = python_value.dtype.itemsize 
    strides = []
    for s in python_value.strides:
      strides.append(specialization_const(s/elt_size)) 
    return abstract_array(strides)
  elif t is tuple:
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
def specialize_abstract_values(fn, abstract_values):
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


def specialize(fn, python_values):
  return specialize_abstract_values(fn,  from_python_list(python_values))
  