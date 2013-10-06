
import numpy as np

# partial mapping, since ctypes doesn't support
# complex numbers
bool8 = np.dtype('bool8')

int8 = np.dtype('int8')
uint8 = np.dtype('uint8')

int16 = np.dtype('int16')
uint16 = np.dtype('uint16')

int32 = np.dtype('int32')
uint32 = np.dtype('uint32')

int64 = np.dtype('int64')
uint64 = np.dtype('uint64')
  
  
float32 = np.dtype('float32')
float64 = np.dtype('float64')
complex64 = np.dtype('complex64')
complex128 = np.dtype('complex128')
  
def is_float(dtype):
  return dtype.type in np.sctypes['float']

def is_signed(dtype):
  return dtype.type in np.sctypes['int']
  
def is_unsigned(dtype):
  return dtype.type in np.sctypes['uint']

def is_complex(dtype):
  return dtype.type in np.sctypes['complex']
   
def is_bool(dtype):
  return dtype == np.bool8
   
def is_int(dtype):
  return is_bool(dtype) or is_signed(dtype) or is_unsigned(dtype)
  

import ctypes 
_to_ctypes = {
  bool8 : ctypes.c_bool, 
  
  int8  : ctypes.c_int8, 
  uint8 : ctypes.c_uint8, 
  
  int16 : ctypes.c_int16, 
  uint16 : ctypes.c_uint16,
  
  int32 : ctypes.c_int32, 
  uint32 : ctypes.c_uint32,
  
  int64 : ctypes.c_int64, 
  uint64 : ctypes.c_uint64, 
  
  float32 : ctypes.c_float, 
  float64 :  ctypes.c_double, 
}

def to_ctypes(dtype):
  """
  Give the ctypes representation for each numpy scalar type. 
  Beware that complex numbers have no assumed representation 
  and thus aren't valid arguments to this function. 
  """
  if dtype in _to_ctypes:
    return _to_ctypes[dtype]
  else:
    raise RuntimeError("No conversion from %s to ctypes" % dtype)
  
