from parakeet.ndtypes import Int8, Int16, Int32, Float32, Float64
from parakeet.ndtypes import is_scalar_subtype
from parakeet.testing_helpers import run_local_tests 

def test_int32_subtypes():
  assert is_scalar_subtype(Int8, Int32)
  assert not is_scalar_subtype(Int32, Int8)
  assert is_scalar_subtype(Int32, Float32)
  assert not is_scalar_subtype(Float32, Int32)

def test_float32_subtypes():
  assert is_scalar_subtype(Int8, Float32)
  assert not is_scalar_subtype(Float32, Int8)
  assert is_scalar_subtype(Float32, Float64)
  assert not is_scalar_subtype(Float64, Float32)

if __name__ == '__main__':
  run_local_tests()
