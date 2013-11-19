import sys
import time 
import numpy as np

from nose.tools import nottest

from dsltools.testing_helpers import eq, expect_eq, run_local_tests
  
from . import (type_conv, type_inference, config, 
              specialize, translate_function_value,
              find_broken_transform,
              run_untyped_fn, 
              run_typed_fn, 
              run_python_fn)




def _copy(x):
  if isinstance(x, np.ndarray):
    return x.copy()
  else:
    return x

def _copy_list(xs):
  return [_copy(x) for x in xs]


def expect(fn, args, expected, msg = None, valid_types = None):
  """
  Helper function used for testing, assert that Parakeet evaluates given code to
  the correct result
  """
  if hasattr(expected, 'dtype') and expected.dtype == 'float16':
    expected = expected.astype('float32')

  untyped_fn = translate_function_value(fn)
  
  available_backends = ['interp', 'c', 'openmp']

  #import cuda_backend 
  #if cuda_backend.device_info.has_gpu():
  #  available_backends.append('cuda')
     

  for backend in available_backends: #available_backends:
    try: 
      result = run_untyped_fn(untyped_fn, _copy_list(args), backend = backend)
      
    except: 
      if config.testing_find_broken_transform: 
        find_broken_transform(fn, args, expected)
      raise
   
    label = "backend=%s, inputs = %s" % (backend, ", ".join(str(arg) for arg in args))

    if msg is not None:
      label += ": " + str(msg)
    
    try: 
      #if hasattr(result, 'flags'):
      #  print result.flags
      expect_eq(result, expected, label)
    except: 
      if config.testing_find_broken_transform: 
        find_broken_transform(fn, args, expected)
      raise 
    
    if valid_types is not None:
      if not isinstance(valid_types, (tuple, list)):
        valid_types = [valid_types]
      assert type(result) in valid_types, \
        "Expected result to have type in %s but got %s" % (valid_types, type(result))
  
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

def assert_eq_arrays(numpy_result, parakeet_result, test_name = None):
  if test_name is None:
    msg = ""
  else:
    msg = "[%s] " % test_name
  assert type(numpy_result) == type(parakeet_result), \
    "%sExpected type %s but got %s" % (msg, type(numpy_result), type(parakeet_result))
  if hasattr(numpy_result, 'shape'):
    assert hasattr(parakeet_result, 'shape')
    assert numpy_result.shape == parakeet_result.shape, \
      "%sExpected shape %s but got %s" % (msg, numpy_result.shape, parakeet_result.shape)
    assert eq(numpy_result, parakeet_result), \
      "%sExpected value %s but got %s" % (msg, numpy_result, parakeet_result)
    
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
