import numpy as np 

from llvm.ee import GenericValue 

import llvm_types 
from llvm_context import  opt_context
import core_types
import type_conv  
import ctypes


  
def python_to_generic_value(x, t):
  ctypes_repr = type_conv.ctypes_repr(t)
  llvm_repr = llvm_types.ctypes_struct_to_lltype(ctypes_repr, t.node_type())
  
  if isinstance(t, core_types.FloatT):
    return GenericValue.real(llvm_repr, x)
  elif isinstance(t, core_types.IntT):
    return GenericValue.int(llvm_repr, x)
  elif isinstance(t, core_types.PtrT):
    return GenericValue.pointer(x)
  else:
    ctypes_obj = type_conv.from_python(x)
    return GenericValue.pointer(ctypes.addressof(ctypes_obj))



def generic_value_to_python(gv, t):
  ctypes_repr = type_conv.ctypes_repr(t)
  
  if isinstance(t, core_types.IntT):
    x = gv.as_int()
  elif isinstance(t, core_types.FloatT):
    llvm_t = llvm_types.ctypes_scalar_to_lltype(ctypes_repr)
    
    x = gv.as_real(gv, llvm_t)
    return t.dtype.type(x)
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
      # if the function's calling conventions expect a pointer to the returned value
      # then we construct it via ctypes before calling into LLVM 
      return_t = self.parakeet_fn.return_type
      ctypes_repr = type_conv.ctypes_repr(return_t)
      return_obj = ctypes_repr()
      return_obj_addr = ctypes.addressof(return_obj)
      gv_return = GenericValue.pointer(return_obj_addr)    
      self.exec_engine.run_function(self.llvm_fn, [gv_return] + gv_inputs)
      return type_conv.to_python(return_obj, return_t)
    else:
      gv_return = self.exec_engine.run_function(self.llvm_fn, gv_inputs)
      return generic_value_to_python(gv_return, self.parakeet_fn.return_type)
    