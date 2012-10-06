import numpy as np 
import ptype

from llvm.ee import *
from llvm.ee import GenericValue 

from llvm_types import dtype_to_lltype, int8_t, float32_t, float64_t 

def scalar_to_generic_value(x, t):
  if isinstance(t, ptype.FloatT):
    return GenericValue.real(dtype_to_lltype(t.dtype), x)
  elif t == ptype.Bool:
    return GenericValue.int(int8_t, x)
  else:
    assert isinstance(t, ptype.IntT)
    # assume it's an integer
    return GenericValue.int(dtype_to_lltype(t.dtype), x)
  
def python_to_generic_value(x, t):
  if isinstance(t, ptype.ScalarT):
    return scalar_to_generic_value(x,t)
  elif isinstance(t, ptype.TupleT):
    raise RuntimeError("I CAN HAZ TUPLE?")
  
def generic_value_to_scalar(gv, t):
  assert isinstance(t, ptype.ScalarT), "Expected %s to be scalar" % t
  if isinstance(t, ptype.IntT):
    x = gv.as_int()
  else:
    assert isinstance(t, ptype.FloatT)
    x = gv.as_real(dtype_to_lltype(t.dtype))
  return t.dtype.type(x)
  

def generic_value_to_python(gv, t):
  if isinstance(t, ptype.ScalarT):
    return generic_value_to_scalar(gv, t)
  addr = gv.as_pointer()
   
  if isinstance(t, ptype.TupleT):
    elts = []
    for elt_t in t.elt_types:
      elt_ptr = addr 
      elt_gv = GenericValue.pointer(elt_ptr)
      elts.append(generic_value_to_python(elt_gv, elt_t))
    return tuple(elts)
  else:
    raise RuntimeError("Don't know how to convert values of %s to python" % t)
