
from .. ndtypes import (Int8, Int16, Int32, Int64, 
                        UInt8, UInt16, UInt32, UInt64,
                        Float32, Float64, Bool) 

from .. frontend import macro, jit 
from .. syntax import Cast 

@macro 
def int8(x):
  return Cast(x, type = Int8) 

@macro 
def int16(x):
  return Cast(x, type = Int16) 

@macro 
def int32(x):
  return Cast(x, type = Int32) 

@macro 
def int64(x):
  return Cast(x, type = Int64) 


@macro 
def uint8(x):
  return Cast(x, type = UInt8) 

@macro 
def uint16(x):
  return Cast(x, type = UInt16) 

@macro 
def uint32(x):
  return Cast(x, type = UInt32) 

@macro 
def uint64(x):
  return Cast(x, type = UInt64)

uint = uint64 

@macro 
def float32(x):
  return Cast(x, type = Float32)

@macro 
def float64(x):
  return Cast(x, type = Float64)

@macro 
def bool(x):
  return Cast(x, type = Bool)
