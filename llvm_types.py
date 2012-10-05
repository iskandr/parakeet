
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

ptr_int8_t = lltype.pointer(int8_t)
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

tuple_struct_cache = {}

def tuple_type(parakeet_elt_types):
  parakeet_elt_types = tuple(parakeet_elt_types)
  if parakeet_elt_types in tuple_struct_cache:
    return tuple_struct_cache[parakeet_elt_types]
  else:  
    n = len(parakeet_elt_types)
    llvm_elt_types = [llvm_ref_type(t) for t in parakeet_elt_types]
    llvm_t = lltype.struct(llvm_elt_types, "tuple%d" % n)
    tuple_struct_cache[parakeet_elt_types] = llvm_t
    return llvm_t

# unlike conventional closures, we don't carry around a function pointer 
# but rather call different typed specializations in different argument-type contexts
# so the first element is a number uniquely identifying the untyped function and its
# partially applied argument types
closure_id_t = int64_t
opaque_closure_t = lltype.struct([closure_id_t, ptr_int8_t], "opaque_closure")

def closure_type(parakeet_arg_types):
  arg_tuple_t = tuple_type(parakeet_arg_types)
  return lltype.struct([closure_id_t, lltype.pointer(arg_tuple_t)])


def llvm_value_type(t):
  if isinstance(t, ptype.ScalarT):
    return dtype_to_lltype(t.dtype)
  elif isinstance(t, ptype.TupleT):
    return tuple_type(t.elt_types)
    
  elif isinstance(t, ptype.ClosureT):
    return closure_type(t.args)
  
  elif isinstance(t, ptype.ClosureSet):
    return opaque_closure_t  
  else:
    elt_t = dtype_to_lltype(t.dtype)
    arr_t = lltype.pointer(elt_t)
    # arrays are a pointer to their data and
    # pointers to shape and strides arrays
    return lltype.struct([arr_t, ptr_int64_t, ptr_int64_t])

def llvm_ref_type(t):
  llvm_value_t = llvm_value_type(t)
  if isinstance(t, ptype.ScalarT):
    return llvm_value_t
  else:
    return lltype.pointer(llvm_value_t)

# we allocate heap slots for output scalars before entering the
# function
def to_llvm_output_type(t):
  llvm_type = llvm_value_type(t)
  if isinstance(t, ptype.ScalarT):
    return lltype.pointer(llvm_type)
  else:
    return llvm_type
  


def convert_float(llvm_value, old_ptype, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  if old_ptype == new_ptype:
    return llvm_value
  
  dest_llvm_type = llvm_value_type(new_ptype)
  dest_name = "%s.cast_%s" % (llvm_value.name, new_ptype)
  
  
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
    return builder.fcmp(llcore.FCMP_ONE, llcore.Constant(llvm_value_type(old_ptype), 0.0))



def convert_signed(llvm_value, old_ptype, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  if old_ptype == new_ptype:
    return llvm_value
  
  dest_llvm_type = llvm_value_type(new_ptype)
  dest_name = "%s.cast_%s" % (llvm_value.name, new_ptype)
  
  
  if isinstance(new_ptype, ptype.FloatT):
    return builder.sitofp(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, ptype.BoolT):
    return builder.fcmp(llcore.ICMP_NE, llcore.Constant(llvm_value_type(old_ptype), 0))
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
  
  dest_llvm_type = llvm_value_type(new_ptype)
  dest_name = "%s.cast_%s" % (llvm_value.name, new_ptype)
  
  if isinstance(new_ptype, ptype.FloatT):
    return builder.uitofp(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, ptype.BoolT):
    return builder.fcmp(llcore.ICMP_NE, llcore.Constant(llvm_value_type(old_ptype), 0))
  else:
    assert isinstance(new_ptype, ptype.SignedT) or isinstance(new_ptype, ptype.UnsignedT)
  
    if old_ptype.nbytes() == new_ptype.nbytes():
      return builder.bitcast(llvm_value, dest_llvm_type, dest_name)
    elif old_ptype.nbytes() < new_ptype.nbytes():
      return builder.zext(llvm_value, dest_llvm_type, dest_name)
    else:
      return builder.trunc(llvm_value, dest_llvm_type, dest_name)
  
def convert_bool(llvm_value, new_ptype, builder):
  dest_llvm_type = llvm_value_type(new_ptype)
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
         
     
