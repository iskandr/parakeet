
import llvm.core as llcore 
from llvm.core import Type as lltype
from core_types import ScalarT, FloatT, BoolT, IntT, UnsignedT, SignedT
from ctypes_repr import to_ctypes  
import ctypes 

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

  
ctypes_scalars_to_llvm_types = {
  ctypes.c_bool : int1_t,
  
  ctypes.c_uint8 : int8_t,  
  ctypes.c_int8 : int8_t,
  
  ctypes.c_uint16 : int16_t,  
  ctypes.c_int16 : int16_t, 
  
  ctypes.c_uint32 : int32_t,
  ctypes.c_int32 : int32_t,
   
  ctypes.c_uint64 : int64_t,
  ctypes.c_int64 : int64_t,
  
  ctypes.c_float : float32_t,
  ctypes.c_double : float64_t, 
}

_struct_cache = {}
def ctypes_struct_to_lltype(S, name = None):
  if not name:
    name = S.__class__.__name__
  
  key = tuple(S._fields_)
  
  if key in _struct_cache:
    return _struct_cache[key]
  else:
    llvm_field_types = []
    for (_, field_type) in S._fields_:
      llvm_field_types.append(ctypes_to_lltype(field_type))
    llvm_struct = lltype.struct(llvm_field_types, name)
    _struct_cache[key] = llvm_struct
    return llvm_struct 
    

ctypes_pointer_class = type(ctypes.POINTER(ctypes.c_int32))
def ctypes_to_lltype(ct, name = None):
  if isinstance(ct, ctypes.Structure):
    return ctypes_struct_to_lltype(ct, name)
  elif isinstance(ct, ctypes_pointer_class):
    return lltype.ptr(ctypes_to_lltype(ct._type_))
  else:
    assert ct in ctypes_scalars_to_llvm_types
    return ctypes_scalars_to_llvm_types[ct]


def llvm_value_type(t):
  return ctypes_to_lltype(to_ctypes(t), t.node_type())

def llvm_ref_type(t):
  llvm_value_t = llvm_value_type(t)
  if isinstance(t, ScalarT):
    return llvm_value_t
  else:
    return lltype.pointer(llvm_value_t)

# we allocate heap slots for output scalars before entering the
# function
def to_llvm_output_type(t):
  llvm_type = llvm_value_type(t)
  if isinstance(t, ScalarT):
    return lltype.pointer(llvm_type)
  else:
    return llvm_type
  

def convert_to_bit(llvm_value, builder):
  llvm_t = llvm_value.type 
  if isinstance(llvm_t, llcore.IntegerType):
    return builder.icmp(llcore.ICMP_NE, llcore.Constant(llvm_t, 0), "ne_zero")
  else:
    return builder.fcmp(llcore.FCMP_ONE, llcore.Constant(llvm_t, 0), "ne_zero")

def convert_from_bit(llvm_value, new_ptype, builder):
  dest_llvm_type = llvm_value_type(new_ptype)
  one = llcore.Constant(dest_llvm_type, 1.0 if isinstance(new_ptype, FloatT) else 1)
  zero = llcore.Constant(dest_llvm_type, 0.0 if isinstance(new_ptype, FloatT) else 0)
  return builder.select(llvm_value, one, zero, "%s.cast.%s" % (llvm_value.name, new_ptype))


def convert_to_bool(llvm_value, old_ptype, builder):
  """
  bools are stored as bytes, if you need to use a boolean
  value for control flow convert it to a bit instead
  """
  bit = convert_to_bit(llvm_value, builder)
  return builder.zext(bit, int8_t, "bool_val")

def convert_from_float(llvm_value, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  dest_llvm_type = llvm_value_type(new_ptype)
  dest_name = "%s.cast_%s" % (llvm_value.name, new_ptype)
  
  
  if isinstance(new_ptype, FloatT):
    nbytes = llvm_value.type.width 
    if nbytes <= new_ptype.nbytes():
      return builder.fpext(llvm_value, dest_llvm_type, dest_name)
    else:
      return builder.fptrunc(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, SignedT):
    return builder.fptosi(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, UnsignedT):
    return builder.fptoui(llvm_value, dest_llvm_type, dest_name)
  else:
    return convert_to_bool(llvm_value, builder)
    



def convert_from_signed(llvm_value, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  
  dest_llvm_type = llvm_value_type(new_ptype)
  dest_name = "%s.cast_%s" % (llvm_value.name, new_ptype)
  
  
  if isinstance(new_ptype, FloatT):
    return builder.sitofp(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, BoolT):
    return convert_to_bool(llvm_value, builder)
  else:
    assert isinstance(new_ptype, IntT)
    nbytes = llvm_value.type.width 
    if nbytes == new_ptype.nbytes():
      return builder.bitcast(llvm_value, dest_llvm_type, dest_name)
    elif nbytes < new_ptype.nbytes():
      return builder.zext(llvm_value, dest_llvm_type, dest_name)
    else:
      return builder.trunc(llvm_value, dest_llvm_type, dest_name)
    

def convert_from_unsigned(llvm_value, new_ptype, builder):
  """Convert from an LLVM float value to some other LLVM scalar type"""
  dest_llvm_type = llvm_value_type(new_ptype)
  dest_name = "%s.cast_%s" % (llvm_value.name, new_ptype)
  
  if isinstance(new_ptype, FloatT):
    return builder.uitofp(llvm_value, dest_llvm_type, dest_name)
  elif isinstance(new_ptype, BoolT):
    return convert_to_bool(llvm_value, builder)
  else:
    assert isinstance(new_ptype, IntT)
    nbytes = llvm_value.type.width 
    if nbytes == new_ptype.nbytes():
      return builder.bitcast(llvm_value, dest_llvm_type, dest_name)
    elif nbytes < new_ptype.nbytes():
      return builder.zext(llvm_value, dest_llvm_type, dest_name)
    else:
      return builder.trunc(llvm_value, dest_llvm_type, dest_name)
  

  
def convert(llvm_value, old_ptype, new_ptype, builder):
  """
  Given an LLVM value and two parakeet types, generate the instruction
  to perform the conversion
  """
  if old_ptype == new_ptype:
    return llvm_value 
    
  if isinstance(old_ptype, FloatT):    
    return convert_from_float(llvm_value, new_ptype, builder)
  elif isinstance(old_ptype, SignedT):
    return convert_from_signed(llvm_value, new_ptype, builder)
     
  else:
    assert isinstance(old_ptype, (BoolT, UnsignedT))
    return convert_from_unsigned(llvm_value, new_ptype, builder)
  
         
     
