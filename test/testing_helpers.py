import sys
import time 
import numpy as np

from nose.tools import nottest
from treelike.testing_helpers import  run_local_tests, eq, expect_eq
  
import parakeet

from parakeet import (type_conv, type_inference, 
                      specialize, translate_function_value,
                      run_typed_fn, run_python_fn)

def _copy(x):
  if isinstance(x, np.ndarray):
    return x.copy()
  else:
    return x

def _copy_list(xs):
  return [_copy(x) for x in xs]

def expect(fn, args, expected, valid_types = None):
  """
  Helper function used for testing, assert that Parakeet evaluates given code to
  the correct result
  """

  interp_result = run_python_fn(fn, _copy_list(args), backend = "interp")
  expect_eq(interp_result, expected)

  llvm_result = run_python_fn(fn, _copy_list(args), backend="llvm")
  if valid_types is not None:
    if not isinstance(valid_types, (tuple, list)):
      valid_types = [valid_types]
    assert type(llvm_result) in valid_types, \
      "Expected result to have type in %s but got %s" % (valid_types, type(llvm_result))
  expect_eq(llvm_result, expected)
  
def expect_each(parakeet_fn, python_fn, inputs):
  for x in inputs:
    expect(parakeet_fn, [x], python_fn(x))

def expect_allpairs(parakeet_fn, python_fn, inputs):
  for x in inputs:
    for y in inputs:
      expect(parakeet_fn, [x,y], python_fn(x,y))

def return_type(fn, input_types):
  untyped_fundef = translate_function_value(fn)
  closure_args = untyped_fundef.python_nonlocals()
  closure_arg_types = map(type_conv.typeof, closure_args)
  return type_inference.infer_return_type(untyped_fundef,
                                          closure_arg_types + input_types)

def expect_type(fn, input_types, output_type):
  actual = return_type(fn, input_types)
  assert actual == output_type, "Expected type %s, actual %s" % \
                                (output_type, actual)
@nottest
def timed_test(parakeet_fn, parakeet_args, python_fn, 
               python_args = None, min_speedup = None):
  if python_args is None:
    python_args = parakeet_args

  start = time.time()
  _ = parakeet_fn(*parakeet_args)
  parakeet_time_with_comp = time.time() - start 

  start = time.time()
  py_result = python_fn(*python_args)
  py_time = time.time() - start 

  start = time.time()
  parakeet_result = parakeet_fn(*parakeet_args)
  parakeet_time_no_comp = time.time() - start 

  print "Parakeet time (with compilation):", parakeet_time_with_comp
  print "Parakeet time (without compilation):", parakeet_time_no_comp
  print "Python time:", py_time 

  assert eq(parakeet_result, py_result), \
    "Expected %s but got %s" % (py_result, parakeet_result)
  if min_speedup is not None:
    assert py_time / parakeet_time_no_comp > min_speedup, \
        "Parakeet too slow: %.2f slowdown" % (parakeet_time_no_comp/py_time)
