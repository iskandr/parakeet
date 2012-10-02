import ast_conversion 
import ptype 
from ptype import UInt8, Int8, Int32, Int64, Float16, Float32, Float64, Bool
import specialization 


def return_type(fn, input_types):
  untyped_fundef = ast_conversion.translate_function_value(fn)
  return specialization.infer_return_type(untyped_fundef, input_types)
 
def expect_type(fn, input_types, output_type):
  actual = return_type(fn, input_types) 
  assert actual == output_type, "Expected type %s, actual %s" % (output_type, actual)

def add1(x):
  return x + 1

def call_add1(x):
  return add1(x)

def test_add1():
  expect_type(add1, [Int32], Int32)
  expect_type(add1, [Bool], UInt8)
  expect_type(add1, [Int64], Int64)
  expect_type(add1, [Float32], Float32)

def test_call_add1():
  expect_type(call_add1, [Int32], Int32) 
  expect_type(call_add1, [Float32], Float32)
  expect_type(call_add1, [Bool], UInt8)
  expect_type(add1, [Float32], Float32)
  
def branch_return(b):
  if b:
    return 1
  else:
    return 1.0
  
def test_branch_return():
  expect_type(branch_return, [Bool], Float16)

def branch_assign(b):
  if b:
    x = 0
  else:
    x = 1.0
  return x

def test_branch_assign():
  expect_type(branch_assign, [Bool], Float16)
  
def incr_loop(init, count):
  x = init  
  while x < count:
    x = x + 1
  return x

def test_incr_loop():
  expect_type(incr_loop, [Int32, Int32], Int32)
  expect_type(incr_loop, [Float64, Int32], Float64)
  

if __name__ == '__main__':
  for k,v in locals().items():
    if k.startswith('test_'):
      v()
    



