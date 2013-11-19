from parakeet import UInt8, Int8, Int32, Int64, Float32, Float64, Bool
from parakeet.testing_helpers import expect_type, run_local_tests

def add1(x):
  return x + 1

def call_add1(x):
  return add1(x)

def test_add1():
  expect_type(add1, [Int32], Int64)
  expect_type(add1, [Int8], Int64)
  expect_type(add1, [UInt8], Int64)
  expect_type(add1, [Bool], Int64)
  expect_type(add1, [Int64], Int64)
  expect_type(add1, [Float32], Float64)

def test_call_add1():
  expect_type(call_add1, [Int32], Int64)
  expect_type(call_add1, [Float32], Float64)
  expect_type(call_add1, [Bool], Int64)
  expect_type(add1, [Float32], Float64)

def add(x,y):
  return x + y

def test_add_bools():
  """
  test_add_bools:
  Parakeet booleans don't behave like the default Python type but rather like
  numpy.bool8, where (+) == (or) and (*) == (and)
  """
  expect_type(add, [Bool, Bool], Bool)

def branch_return(b):
  if b:
    return 1
  else:
    return 1.0

def test_branch_return():
  expect_type(branch_return, [Bool], Float64)

def branch_assign(b):
  if b:
    x = 0
  else:
    x = 1.0
  return x

def test_branch_assign():
  expect_type(branch_assign, [Bool], Float64)

def incr_loop(init, count):
  x = init
  while x < count:
    x = x + 1
  return x

def test_incr_loop():
  expect_type(incr_loop, [Int32, Int32], Int64)
  expect_type(incr_loop, [Float64, Int32], Float64)

if __name__ == '__main__':
  run_local_tests()
