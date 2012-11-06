import sys
import numpy as np

import interp
from external_api import specialize_and_compile


def run_local_functions(prefix, locals_dict = None):   
  if locals_dict is None:
    last_frame = sys._getframe() 
    locals_dict = last_frame.f_back.f_locals
  
  good = set([])
  bad = set([])
  for k, test in locals_dict.iteritems():
    
    if k.startswith(prefix):
      print "Running %s..." % k
      try:
        test()
        print "\n --- %s passed\n" % k
        good.add(k)
      
      except:
        #traceback.print_tb(sys.exc_info()[2])
        #print sys.exc_info()
        raise 
        #traceback.print_tb()
        #print sys.exc_info()[1]
        print "\n --- %s failed\n" % k
        bad.add(k)
  print "\n%d tests passed: %s\n" % (len(good), ", ".join(good))
  print "%d failed: %s" % (len(bad),", ".join(bad))

def run_local_tests(locals_dict = None):
  if locals_dict is None:
    last_frame = sys._getframe()
    locals_dict = last_frame.f_back.f_locals
  return run_local_functions("test_", locals_dict)
  

def eq(x,y):
  if isinstance(y, np.ndarray):
    return isinstance(x, np.ndarray) and x.shape == y.shape and \
      (np.all(np.ravel(x) == np.ravel(y)) or \
       np.mean(np.ravel(x) - np.ravel(y)) <= 0.000001)
  else:
    return x == y

def copy(x):
  if isinstance(x, np.ndarray):
    return x.copy()
  else:
    return x

def copy_list(xs):
  return [copy(x) for x in xs]

def expect(fn, args, expected):
  """
  Helper function used for testing, assert that Parakeet evaluates given code to
  the correct result
  """
  untyped,  typed, compiled, all_args = specialize_and_compile(fn, args) 
  
  untyped_result = interp.eval_fn(untyped, copy_list(all_args)) 
  assert eq(untyped_result, expected), \
    "Expected %s but untyped fn returned  %s" % (expected, untyped_result)

  typed_result = interp.eval_fn(typed, copy_list(all_args))
  assert eq(typed_result, expected), \
    "Expected %s but typed fn returned %s" % (expected, typed_result)

  llvm_result = compiled(*all_args)
  assert eq(llvm_result, expected), \
    "Expected %s but compiled fn return %s" % (expected, llvm_result)


def expect_each(parakeet_fn, python_fn, inputs):
  for x in inputs:
    expect(parakeet_fn, [x], python_fn(x))
    
def expect_allpairs(parakeet_fn, python_fn, inputs):
  for x in inputs:
    for y in inputs:
      expect(parakeet_fn, [x,y], python_fn(x,y))


import ast_conversion 
import type_inference 
import type_conv 
def return_type(fn, input_types):
  untyped_fundef = ast_conversion.translate_function_value(fn)
  closure_args = untyped_fundef.python_nonlocals()
  closure_arg_types = map(type_conv.typeof, closure_args)
  return type_inference.infer_return_type(untyped_fundef, closure_arg_types + input_types)
 
def expect_type(fn, input_types, output_type):
  actual = return_type(fn, input_types) 
  assert actual == output_type, "Expected type %s, actual %s" % (output_type, actual)
