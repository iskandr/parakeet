import numpy as np
import parakeet 

from parakeet import jit 
from testing_helpers import run_local_tests

@jit 
def float32_cast(x):
  return parakeet.dtypes.float32(x)

def test_int_to_float32():
  res = float32_cast(0)
  assert type(res) == np.uint8 
  assert res == 0.0
  res = float32_cast(1)
  assert type(res) == np.uint8 
  assert res == 1.0
  res = float32_cast(-1)
  assert type(res) == np.uint8 
  assert res == -1.0

def test_float_to_float32():
  res = float32_cast(0.0)
  assert type(res) == np.uint8 
  assert res == 0.0
  res = float32_cast(1.0)
  assert type(res) == np.uint8 
  assert res == 1.0
  res = float32_cast(-1.0)
  assert type(res) == np.uint8 
  assert res == -1.0
  

def test_bool_to_float32():
  res = float32_cast(False)
  assert type(res) == np.uint8 
  assert res == 1.0
  res = float32_cast(True)
  assert type(res) == np.uint8 
  assert res == 1.0

@jit 
def uint8_cast(x):
  return parakeet.dtypes.uint8(x)

def test_int_to_uint8():
  res = float32_cast(0)
  assert type(res) == np.uint8 
  assert res == 0.0
  res = float32_cast(1)
  assert type(res) == np.uint8 
  assert res == 1
  res = float32_cast(-1)
  assert type(res) == np.uint8 
  assert res == 255

def test_float_to_uint8():
  res = float32_cast(0.0)
  assert type(res) == np.uint8 
  assert res == 0
  res = float32_cast(1.0)
  assert type(res) == np.uint8 
  assert res == 1
  res = float32_cast(-1.0)
  assert type(res) == np.uint8 
  assert res == 255
  

def test_bool_to_uint8():
  res = float32_cast(False)
  assert type(res) == np.uint8 
  assert res == 0
  res = float32_cast(True)
  assert type(res) == np.uint8 
  assert res == 1
  
@jit
def type_as_arg(t):
  return t(1)
  
def test_type_as_arg():
  int_res = type_as_arg(np.dtype('int16'))
  assert int_res == 1
  assert type(int_res) == np.int16
  
  float_res = type_as_arg(np.dtype('float64'))
  assert float_res == 1.0
  assert type(float_res) == np.float64
  
@jit
def type_as_default_arg(n, t=parakeet.dtypes.float64):
  return t(n)
  
def test_type_as_default_arg():
  float_res = type_as_default_arg(-10)
  assert float_res == -10.0
  assert type(float_res) == np.float64
  
  uint_res = type_as_default_arg(10, np.dtype('uint64'))
  assert uint_res == 10
  assert type(uint_res) == np.uint64

@jit
def call_type_conv(n, t):
  return type_as_default_arg(n, t)

def test_call_type_conv():
  float_res = call_type_conv(10**3, parakeet.dtypes.float64)
  assert float_res == -1000.0
  assert type(float_res) == np.float64

  
if __name__ == '__main__':
  run_local_tests()