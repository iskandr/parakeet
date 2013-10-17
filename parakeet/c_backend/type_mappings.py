
from ..ndtypes import (UInt8, UInt16, UInt32, UInt64, 
                       Int8, Int16, Int32, Int64, Float32, Float64, NoneType,  
                       Bool, ArrayT, ClosureT, TupleT, SliceT, NoneT, PtrT)

_dtype_mappings = {
  Bool : "NPY_BOOL",
  Int8 : "NPY_INT8", 
  Int16 : "NPY_INT16",
  Int32 : "NPY_INT32",
  Int64 : "NPY_INT64",
  UInt8 : "NPY_UINT8",
  UInt16 : "NPY_UINT16",
  UInt32 : "NPY_UINT32",
  UInt64 : "NPY_UINT64",
  Float32 : "NPY_FLOAT32",
  Float64 : "NPY_FLOAT64"
}
def to_dtype(t):
  if t in _dtype_mappings:
    return _dtype_mappings[t]
  assert False, "Unsupported element type %s" % t
  
_ctype_mappings = {
  Int8:  "int8_t",
  UInt8:  "uint8_t",
  UInt16:  "uint16_t",
  UInt32:  "uint32_t",
  UInt64:  "uint64_t",
  Int16:  "int16_t",
  Int32:  "int32_t",
  Int64:  "int64_t",
  Float32:  "float",
  Float64:  "double",
  NoneType:  "int",
  Bool:  "int8_t",
}

def to_ctype(t):
  if t in _ctype_mappings:
    return _ctype_mappings[t]
  elif isinstance(t, PtrT):
    return "%s*" % to_ctype(t.elt_type)
  elif isinstance(t, (ArrayT, ClosureT, TupleT, SliceT, NoneT)):
    return "PyObject*"
  else:
    assert False, "Unsupported type %s" % t
    