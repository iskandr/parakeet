import ast_conversion 
import ptype 
from ptype import Int32, Int64, Float32, Float64, Bool
import type_analysis 


def return_type(fn, input_types):
  untyped_fundef = ast_conversion.translate_function_value(fn)
  ret_type, _ = type_analysis.infer_types(untyped_fundef, input_types)
  return ret_type 

def expect_type(fn, input_type, output_type):
  actual = return_type(fn, [input_type]) 
  assert actual == output_type, "Expected type %s, actual %s" % (output_type, actual)

def add1(x):
  return x + 1


def test_add1_int32():
  """ The constant 1 is an int64, so the return type should be int64"""
  expect_type(add1, Int32, Int64)

def test_add1_int64():
  expect_type(add1, Int64, Int64)

def test_add1_float32():
  expect_type(add1, Float32, Float64)

def test_add1_bool():
  expect_type(add1, Bool, Int64)
  