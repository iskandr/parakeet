import numpy as np
import parakeet 
import testing_helpers

from parakeet import jit 
from testing_helpers import eq, expect, run_local_tests

@jit 
def float32_cast(x):
  return parakeet.dtypes.float32(x)

def test_int_to_float32(x):
  res = float32_cast(0)
  assert type(res) == np.float32 
  assert res == 0.0
  res = float32_cast(1)
  assert type(res) == np.float32 
  assert res == 1.0
  res = float32_cast(-1)
  assert type(res) == np.float32 
  assert res == -1.0

def test_float_to_float32(x):
  res = float32_cast(0.0)
  assert type(res) == np.float32 
  assert res == 0.0
  res = float32_cast(1.0)
  assert type(res) == np.float32 
  assert res == 1.0
  res = float32_cast(-1.0)
  assert type(res) == np.float32 
  assert res == -1.0
  

def test_bool_to_float32(x):
  res = float32_cast(False)
  assert type(res) == np.float32 
  assert res == 1.0
  res = float32_cast(True)
  assert type(res) == np.float32 
  assert res == 1.0

@jit 
def uint8_cast(x):
  return parakeet.dtypes.uint8(x)


def test_int_to_uint8(x):
  res = float32_cast(0)
  assert type(res) == np.float32 
  assert res == 0.0
  res = float32_cast(1)
  assert type(res) == np.float32 
  assert res == 1.0
  res = float32_cast(-1)
  assert type(res) == np.float32 
  assert res == -1.0

def test_float_to_uint8(x):
  res = float32_cast(0.0)
  assert type(res) == np.float32 
  assert res == 0.0
  res = float32_cast(1.0)
  assert type(res) == np.float32 
  assert res == 1.0
  res = float32_cast(-1.0)
  assert type(res) == np.float32 
  assert res == -1.0
  

def test_bool_to_uint8(x):
  res = float32_cast(False)
  assert type(res) == np.float32 
  assert res == 1.0
  res = float32_cast(True)
  assert type(res) == np.float32 
  assert res == 1.0