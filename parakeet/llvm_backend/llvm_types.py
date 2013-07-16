import ctypes
import llvm.core as llcore

from llvm.core import Type as lltype

from .. ndtypes import ScalarT, PtrT, NoneT

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

def nbytes(t):
  if t.kind == llcore.TYPE_FLOAT:
    return 4
  elif t.kind == llcore.TYPE_DOUBLE:
    return 8
  else:
    return t.width / 8

def is_scalar(llvm_t):
  return isinstance(llvm_t, llcore.IntegerType) or \
         llvm_t in (float32_t, float64_t)

_ctypes_scalars_to_llvm_types = {
  ctypes.c_bool : int8_t,
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

PyCPointerType = type(ctypes.POINTER(ctypes.c_int32))
PyCStructType = type(ctypes.Structure)

_struct_cache = {}
def ctypes_struct_to_lltype(S, name = None):
  if not name:
    name = S.__class__.__name__

  key = tuple(S._fields_)

  if key in _struct_cache:
    return _struct_cache[key]
  else:
    llvm_field_types = [ctypes_to_lltype(field_type)
                        for (_, field_type) in S._fields_]
    llvm_struct = lltype.struct(llvm_field_types, name)
    _struct_cache[key] = llvm_struct
    return llvm_struct

def ctypes_scalar_to_lltype(ct):
  assert ct in _ctypes_scalars_to_llvm_types, \
      "%s isn't a convertible to an LLVM scalar type" % ct
  return _ctypes_scalars_to_llvm_types[ct]

def ctypes_to_lltype(ctypes_repr, name = None):
  if type(ctypes_repr) == PyCStructType:
    return ctypes_struct_to_lltype(ctypes_repr, name)
  elif type(ctypes_repr) == PyCPointerType:
    elt_t = ctypes_repr._type_
    if elt_t == ctypes.c_bool:
      return lltype.pointer(int8_t)
    else:
      return lltype.pointer(ctypes_to_lltype(elt_t))
  else:
    return ctypes_scalar_to_lltype(ctypes_repr)

def llvm_value_type(t):
  return ctypes_to_lltype(t.ctypes_repr, t.node_type())

def llvm_ref_type(t):
  llvm_value_t = llvm_value_type(t)
  if isinstance(t, (PtrT, ScalarT, NoneT)):
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
