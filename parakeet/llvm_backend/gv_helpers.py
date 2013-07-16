import ctypes 

from llvm.ee import GenericValue 

from .. ndtypes import FloatT, SignedT, IntT, NoneT, PtrT    
from .. ndtypes import type_conv 
import llvm_types 

def python_to_generic_value(x, t):
  if isinstance(t, FloatT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.real(llvm_t, x)
  elif isinstance(t, SignedT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.int_signed(llvm_t, x)
  elif isinstance(t, IntT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.int(llvm_t, x)
  elif isinstance(t, PtrT):
    return GenericValue.pointer(x)
  else:
    ctypes_obj = type_conv.from_python(x)
    return GenericValue.pointer(ctypes.addressof(ctypes_obj))

def ctypes_to_generic_value(cval, t):
  if isinstance(t, FloatT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.real(llvm_t, cval.value)
  elif isinstance(t, SignedT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.int_signed(llvm_t, cval.value)
  elif isinstance(t, IntT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.int(llvm_t, cval.value)
  elif isinstance(t, NoneT):
    return GenericValue.int(llvm_types.int64_t, 0)
  elif isinstance(t, PtrT):
    return GenericValue.pointer(ctypes.addressof(cval.contents))
  else:
    return GenericValue.pointer(ctypes.addressof(cval))

def generic_value_to_python(gv, t):
  if isinstance(t, SignedT):
    return t.dtype.type(gv.as_int_signed() )
  elif isinstance(t, IntT):
    return t.dtype.type( gv.as_int() )
  elif isinstance(t, FloatT):
    llvm_t = llvm_types.ctypes_scalar_to_lltype(t.ctypes_repr)
    return t.dtype.type(gv.as_real(llvm_t))
  elif isinstance(t, NoneT):
    return None
  else:
    addr = gv.as_pointer()
    struct = t.ctypes_repr.from_address(addr)
    return t.to_python(struct)