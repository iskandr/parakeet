import ast_conversion 
import parakeet_types
from parakeet_types import Int32, Float32, Float64, Bool
import type_inference


def return_type(fn, input_types):
  untyped_fundef = ast_conversion.translate_function_value(fn)
  return type_inference.infer_return_type(untyped_fundef, [input_types])
  
def expect_type(fn, input_type, output_type):
  actual = return_type(fn, [input_type]) 
  assert actual == output_type, "Expected type %s, actual %s" % (output_type, actual)

def add1(x):
  return x + 1

def test_add_float():
  expect_type(add1, )
  assert return_type(add1, parakeet_types.Int32, parakeet_types.Int32) == parakeet  