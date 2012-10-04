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
  
def generic_value_to_scalar(gv, t):
  assert isinstance(t, ptype.ScalarT), "Expected %s to be scalar" % t
  if isinstance(t, ptype.IntT):
    x = gv.as_int()
  else:
    assert isinstance(t, ptype.FloatT)
    x = gv.as_real(dtype_to_lltype(t.dtype))
  return t.dtype.type(x)
  
  #if isinstance(x, int) or isinstance(x, long):
  #  return gv.int(int64_t, x)
  #elif isinstance(x, float):
  #  return gv.real(float64_t, x)
  #elif isinstance(x, bool):
  #  return gv.int(int8_t, x)
  # if it's a numpy scalar integer
  #elif isinstance(x, np.integer):
  #  return gv.int(dtype_to_lltype(x.dtype()), x)
  #elif isinstance(x, np.floating):
  #  return gv.real(dtype_to_lltype(x.dtype()), x)
  #else:
  #  raise RuntimeError("Don't know how to convert value " + str(x))
"""
# Given a list of Python values, return a list of generic values
# which will be understood by the LLVM execution engine
# The returned list might possibly be longer since array arguments
# are passed in as the data pointer followed by shape and strides pointers
#def convert_args_to_generic_values(python_values):
"""

#def make_tuple_buffer(elts):
#  n = len(elts)
#  arr = np.zeros(n, dtype=np.int64)
#  for i in xrange(n):
    
