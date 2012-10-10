import numpy as np 

from llvm.ee import GenericValue 

from llvm_types import dtype_to_lltype, int8_t, float32_t, float64_t 
from llvm_context import  opt_context
import ptype 
import ctypes

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
  else:
    ctypes_struct = t.to_ctypes(x)
    return GenericValue.pointer(ctypes.addressof(ctypes_struct))
  
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
  else:
    addr = gv.as_pointer()
    ptr_t = ctypes.POINTER(t.ctypes_repr)
    ptr = ptr_t.from_address(addr)
    struct = ptr.contents
    return t.from_ctypes(struct)
    
    
class CompiledFn:
  def __init__(self, llvm_fn, parakeet_fn, exec_engine = opt_context.exec_engine, sret = True):
    self.llvm_fn = llvm_fn
    self.parakeet_fn = parakeet_fn
    self.exec_engine = exec_engine
    self.sret = sret # calling conventions 
  
  def __call__(self, *args):
    # calling conventions are that output must be preallocated by the caller'
    gv_inputs = []
          
    for (v, t) in zip(args, self.parakeet_fn.input_types):
      if np.isscalar(v):
        gv_inputs.append(python_to_generic_value(v, t))
      else:
        assert False, (v,t)
    
    if self.sret:
      return_t = self.parakeet_fn.return_type
      return_obj = return_t.ctypes_repr()
      return_obj_addr = ctypes.addressof(return_obj)
      gv_return = GenericValue.pointer(return_obj_addr)
      self.exec_engine.run_function(self.llvm_fn, [gv_return] + gv_inputs)
      return return_t.from_ctypes(return_obj)
    else:
      gv_return = self.exec_engine.run_function(self.llvm_fn, gv_inputs)
      return generic_value_to_python(gv_return, self.parakeet_fn.return_type)
    