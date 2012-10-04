
import llvm.core as llcore 
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
  if isinstance(t, ptype.ScalarT):
    return dtype_to_lltype(t.dtype)
  elif isinstance(t, ptype.TupleT):
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
  if isinstance(t, ptype.ScalarT):
    return lltype.pointer(llvm_type)
  else:
    return llvm_type
  


def convert_float(llvm_value, old_ptype, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  if old_ptype == new_ptype:
    return llvm_value
  
  dest_llvm_type = to_lltype(new_ptype)
  dest_name = "%s.cast_%s" % new_ptype
  
  if isinstance(new_ptype, ptype.FloatT):
    if old_ptype.nbytes() <= new_ptype.nbytes():
      return builder.fpext(llvm_value, dest_llvm_type, dest_name)
    else:
      return builder.fptrunc(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, ptype.SignedT):
    return builder.fptosi(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, ptype.UnsignedT):
    return builder.fptoui(llvm_value, dest_llvm_type, dest_name)
  else:
    assert isinstance(new_ptype, ptype.BoolT), \
      "Unexpected type %s when casting from %s" % (new_ptype, old_ptype)
    # float->bool is just a check whether it's != 0 
    return builder.fcmp(llcore.FCMP_ONE, llcore.Constant(to_lltype(old_ptype), 0.0))



def convert_signed(llvm_value, old_ptype, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  if old_ptype == new_ptype:
    return llvm_value
  
  dest_llvm_type = to_lltype(new_ptype)
  dest_name = "%s.cast.%s" % new_ptype
  
  if isinstance(new_ptype, ptype.FloatT):
    return builder.sitofp(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, ptype.BoolT):
    return builder.fcmp(llcore.ICMP_NE, llcore.Constant(to_lltype(old_ptype), 0))
  else:
    assert isinstance(new_ptype, ptype.SignedT) or isinstance(new_ptype, ptype.UnsignedT)
  
    if old_ptype.nbytes() == new_ptype.nbytes():
      return builder.bitcast(llvm_value, dest_llvm_type, dest_name)
    elif old_ptype.nbytes() < new_ptype.nbytes():
      return builder.zext(llvm_value, dest_llvm_type, dest_name)
    else:
      return builder.trunc(llvm_value, dest_llvm_type, dest_name)
    

def convert_unsigned(llvm_value, old_ptype, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  if old_ptype == new_ptype:
    return llvm_value
  
  dest_llvm_type = to_lltype(new_ptype)
  dest_name = "%s.cast_%s" % new_ptype
  
  if isinstance(new_ptype, ptype.FloatT):
    return builder.uitofp(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, ptype.BoolT):
    return builder.fcmp(llcore.ICMP_NE, llcore.Constant(to_lltype(old_ptype), 0))
  else:
    assert isinstance(new_ptype, ptype.SignedT) or isinstance(new_ptype, ptype.UnsignedT)
  
    if old_ptype.nbytes() == new_ptype.nbytes():
      return builder.bitcast(llvm_value, dest_llvm_type, dest_name)
    elif old_ptype.nbytes() < new_ptype.nbytes():
      return builder.zext(llvm_value, dest_llvm_type, dest_name)
    else:
      return builder.trunc(llvm_value, dest_llvm_type, dest_name)
  
def convert_bool(llvm_value, new_ptype, builder):
  dest_llvm_type = to_lltype(new_ptype)
  one = llcore.Constant(dest_llvm_type, 1.0 if isinstance(new_ptype, ptype.FloatT) else 1)
  zero = llcore.Constant(dest_llvm_type, 0.0 if isinstance(new_ptype, ptype.FloatT) else 0)
  return builder.select(llvm_value, one, zero, "%s.cast.%s" % (llvm_value.name, new_ptype))

  
def convert(llvm_value, old_ptype, new_ptype, builder):
  """
  Given an LLVM value and two parakeet types, generate the instruction
  to perform the conversion
  """
  if old_ptype == new_ptype:
    return llvm_value 
    
  if isinstance(old_ptype, ptype.FloatT):    
    return convert_float(llvm_value, old_ptype, new_ptype, builder)
  elif isinstance(old_ptype, ptype.SignedT):
    return convert_signed(llvm_value, old_ptype, new_ptype, builder)
     
  elif isinstance(old_ptype, ptype.UnsignedT):
    return convert_unsigned(llvm_value, old_ptype, new_ptype, builder)
  else:
    assert old_ptype == ptype.Bool
    return convert_bool(llvm_value, new_ptype, builder)
         
     
