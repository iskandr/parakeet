import ast_conversion 
import ptype 
from ptype import Int32, Int64, Float32, Float64, Bool
import type_analysis 


def return_type(fn, input_types):
  untyped_fundef = ast_conversion.translate_function_value(fn)
  ret_type, _ = type_analysis.infer_types(untyped_fundef, input_types)
  return ret_type 

def expect_type(fn, input_types, output_type):
  actual = return_type(fn, input_types) 
  assert actual == output_type, "Expected type %s, actual %s" % (output_type, actual)

def add1(x):
  return x + 1

def call_add1(x):
  return add1(x)

def test_add1():
  """ The constant 1 is an int64, so the return type should be int64"""
  expect_type(add1, [Int32], Int64)
  expect_type(add1, [Bool], Int64)
  expect_type(add1, [Int64], Int64)
  expect_type(add1, [Float32], Float64)

def test_call_add1():
  expect_type(call_add1, [Int32], Int64) 
  expect_type(call_add1, [Float32], Float64)
  expect_type(call_add1, [Bool], Int64)
  expect_type(add1, [Float32], Float64)
  
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



