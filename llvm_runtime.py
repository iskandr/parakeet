import numpy as np 
import ptype

from llvm.ee import *
from llvm.ee import GenericValue as gv

from llvm_types import dtype_to_lltype, int8_t

def scalar_to_generic_value(x, t):
  if t.is_float():
    return gv.real(dtype_to_lltype(t.dtype), x)
  elif t.is_bool():
    return gv.int(int8_t, x)
  else:
    assert t.is_int()
    # assume it's an integer
    return gv.int(dtype_to_lltype(t.dtype), x)
  
  
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
    

def run(llvm_fn_info, python_values):
  inputs = []  
  for (v, t) in zip(python_values, llvm_fn_info.parakeet_input_types):
    # TODO: make arrays just a tuple! 
    if isinstance(v, np.ndarray):
      
      assert isinstance(t, ptype.Array)
      # convert the shape and strides tuples
      # into uniform arrays so we can pass them
      # as an int* rather than PyObject*
      shape = np.array(v.shape)
      strides = np.array(v.strides)
      # the generated LLVM code will expect
      # the shape and strides of an array to simply
      # be passed after the array's data pointer
      inputs.append(shape.data)
      inputs.append(strides.data)
    elif np.isscalar(v):
      return scalar_to_generic_value(v, t)
    else:
      assert isinstance(v, tuple)
  
# NOW ACTUALLY RUN SOMETHING! 