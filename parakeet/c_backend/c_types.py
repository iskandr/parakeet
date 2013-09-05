
from ..ndtypes import (UInt8, UInt16, UInt32, UInt64, 
                       Int8, Int16, Int32, Int64, Float32, Float64, NoneType,  
                       TupleT, ScalarT, NoneT, ArrayT, SliceT, Type, 
                       ClosureT, PtrT)




def to_dtype(t):
  if t == Int8: return "NPY_INT8"
  elif t == Int16: return "NPY_INT16"
  elif t == Int32: return "NPY_INT32"
  elif t == Int64: return "NPY_INT64"
  elif t == Float32: return "NPY_FLOAT32"
  elif t == Float64: return "NPY_FLOAT64"
  else:
    assert False, "Unsupported element type %s" % t
  
def to_ctype(t):
  if t == Int8: return "int8_t"
  elif t == UInt8: return "uint8_t"
  elif t == UInt16: return "uint16_t"
  elif t == UInt32: return "uint32_t"
  elif t == UInt64: return "uint64_t"
  elif t == Int16: return "int16_t"
  elif t == Int32: return "int32_t"
  elif t == Int64: return "int64_t"
  elif t == Float32: return "float"
  elif t == Float64: return "double"
  elif t == NoneType: return "void"
  
  #elif isinstance(t, ArrayT):
  #  return "PyArrayObject*"
  #elif isinstance(t, (ClosureT, TupleT)):
  #  return "PyTupleObject*"
  #elif isinstance(t, SliceT):
  #  return "PySliceObject*"
  #elif isinstance(t, NoneT):
  #  return "PyObject*"
  elif isinstance(t, PtrT):
    return "%s*" % to_ctype(t.elt_type)
  elif isinstance(t, (ArrayT, ClosureT, TupleT, SliceT, NoneT)):
    return "PyObject*"
  else:
    assert False, "Unsupported type %s" % t
    
  
class BoxedNumberT(Type):
  """
  When creating wrappers which accept pyobjects, let's pretend all numbers come boxed
  """
  pass 