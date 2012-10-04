
from llvm.core import Type as lltype
import ptype 
import numpy as np 

void_t = lltype.void()
int1_t = lltype.int(1)
int8_t = lltype.int(8)
int16_t = lltype.int(16)
int32_t = lltype.int(32)
int64_t = lltype.int(64)
float32_t = lltype.float()
float64_t = lltype.double()
float128_t = lltype.fp128()

ptr_int32_t = lltype.pointer(int32_t)
ptr_int64_t = lltype.pointer(int64_t)

dtype_to_llvm_types = {
  np.dtype('bool') : int1_t, 
  np.dtype('int8') : int8_t,
  np.dtype('uint8') : int8_t,
  np.dtype('uint16') : int16_t, 
  np.dtype('int16') : int16_t,
  np.dtype('uint32') : int32_t, 
  np.dtype('int32') : int32_t,
  np.dtype('uint64') : int64_t, 
  np.dtype('int64') : int64_t,
  np.dtype('float16') : float32_t, 
  np.dtype('float32') : float32_t,
  np.dtype('float64') : float64_t,
}

def dtype_to_lltype(dt):
  return dtype_to_llvm_types[dt]

def to_lltype(t):
  if isinstance(t, ptype.Scalar):
    return dtype_to_lltype(t.dtype)
  elif isinstance(t, ptype.Tuple):
    llvm_elt_types = map(to_lltype, t.elt_types)
    return lltype.struct(llvm_elt_types)
  else:
    elt_t = dtype_to_lltype(t.dtype)
    arr_t = lltype.pointer(elt_t)
    # arrays are a pointer to their data and
    # pointers to shape and strides arrays
    return lltype.struct([arr_t, ptr_int64_t, ptr_int64_t])
  

# we allocate heap slots for output scalars before entering the
# function
def to_llvm_output_type(t):
  llvm_type = to_lltype(t)
  if isinstance(t, ptype.Scalar):
    return lltype.pointer(llvm_type)
  else:
    return llvm_type
