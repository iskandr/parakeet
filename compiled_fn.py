import numpy as np 

from llvm.ee import GenericValue 

import llvm_types 
from llvm_context import  opt_context
import core_types
import type_conv  
import ctypes


  
def python_to_generic_value(x, t):

  
  if isinstance(t, core_types.FloatT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.real(llvm_t, x)
  elif isinstance(t, core_types.IntT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.int(llvm_t, x)
  elif isinstance(t, core_types.PtrT):
    return GenericValue.pointer(x)
  else:
    ctypes_obj = type_conv.from_python(x)
    return GenericValue.pointer(ctypes.addressof(ctypes_obj))



def generic_value_to_python(gv, t):
  ctypes_repr = type_conv.ctypes_repr(t)
  
  if isinstance(t, core_types.IntT):
    return t.dtype.type(gv.as_int())
  elif isinstance(t, core_types.FloatT):
    llvm_t = llvm_types.ctypes_scalar_to_lltype(ctypes_repr)
    return t.dtype.type(gv.as_real(llvm_t))
  else:
    print "gv", gv
    addr = gv.as_pointer()
    print "addr", addr
    struct = ctypes_repr.from_address(addr)
    print "struct", struct
    print "fields", struct._fields_ 
    print "elt0", struct.elt0  
    return type_conv.to_python(struct, t)
    
    
class CompiledFn:
  def __init__(self, llvm_fn, parakeet_fn, exec_engine = opt_context.exec_engine, sret = True):
    self.llvm_fn = llvm_fn
    self.parakeet_fn = parakeet_fn
    self.exec_engine = exec_engine
    self.sret = sret # calling conventions 
  
  def __call__(self, *args):
    # calling conventions are that output must be preallocated by the caller'
    gv_inputs = [python_to_generic_value(v, t) for (v,t) in zip(args, self.parakeet_fn.input_types)]
    gv_return = self.exec_engine.run_function(self.llvm_fn, gv_inputs)
    return generic_value_to_python(gv_return, self.parakeet_fn.return_type)
    