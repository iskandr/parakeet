import sys
import time 

import numpy as np
from nose.tools import nottest

import parakeet
from parakeet import ast_conversion, interp, type_conv, type_inference
from parakeet.run_function import specialize_and_compile

def run_local_functions(prefix, locals_dict = None):
  if locals_dict is None:
    last_frame = sys._getframe()
    locals_dict = last_frame.f_back.f_locals

  good = set([])
  # bad = set([])
  for k, test in locals_dict.iteritems():
    if k.startswith(prefix):
      print "Running %s..." % k
      try:
        test()
        print "\n --- %s passed\n" % k
        good.add(k)
      except:
        raise

  print "\n%d tests passed: %s\n" % (len(good), ", ".join(good))
  # print "%d failed: %s" % (len(bad),", ".join(bad))

@nottest
def run_local_tests(locals_dict = None):
  if locals_dict is None:
    last_frame = sys._getframe()
    locals_dict = last_frame.f_back.f_locals
  return run_local_functions("test_", locals_dict)

def eq(x,y):
  if isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
    return False
  if isinstance(y, np.ndarray):
    if isinstance(x, np.ndarray) and x.shape == y.shape:
      err = abs(np.mean(np.ravel(x) - np.ravel(y)))
      m = abs(np.mean(np.ravel(x)))
      if not np.all(np.ravel(x) == np.ravel(y)) and err/m > 0.000001:
        print "err:", err
        print "err/m:", err/m
        return False
      else:
        return True
  else:
    return x == y

def copy(x):
  if isinstance(x, np.ndarray):
    return x.copy()
  else:
    return x

def expect(fn, args, expected):
  """
  Helper function used for testing, assert that Parakeet evaluates given code to
  the correct result
  """

  untyped, typed, compiled, all_args = specialize_and_compile(fn, args)

  untyped_result = interp.eval_fn(untyped, all_args.transform(copy))
  assert eq(untyped_result, expected), \
      "Expected %s but untyped fn returned  %s" % (expected, untyped_result)
  linear_args = untyped.args.linearize_without_defaults(all_args)


  typed_result = interp.eval_fn(typed, map(copy, linear_args))
  assert eq(typed_result, expected), \
      "Expected %s but typed fn returned %s" % (expected, typed_result)

  llvm_result = compiled(*linear_args)
  assert eq(llvm_result, expected), \
      "Expected %s but compiled fn returned %s" % (expected, llvm_result)

def expect_each(parakeet_fn, python_fn, inputs):
  for x in inputs:
    expect(parakeet_fn, [x], python_fn(x))

def expect_allpairs(parakeet_fn, python_fn, inputs):
  for x in inputs:
    for y in inputs:
      expect(parakeet_fn, [x,y], python_fn(x,y))

def return_type(fn, input_types):
  untyped_fundef = ast_conversion.translate_function_value(fn)
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
