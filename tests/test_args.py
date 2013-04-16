from testing_helpers import expect, run_local_tests
import parakeet 
import numpy as np 

def varargs_return(*x):
  return x

def test_varargs_return():
  expect(varargs_return, [1,2], (1,2))

def varargs_add(*x):
  return x[0] + x[1]

def test_varargs_add():
  expect(varargs_add, [1,2], 3)

def call_varargs_add(x,y):
  local_tuple = (x,y)
  return varargs_add(*local_tuple)

def test_call_varargs_add():
  expect(call_varargs_add, [1,2], 3)
  expect(call_varargs_add, [True,2.0], 3.0)

def add_defaults(x = 1, y = 2):
    return x + y

def test_add_defaults():
  expect(add_defaults, [], 3)
  expect(add_defaults, [10], 12)
  expect(add_defaults, [10, 20], 30)
  expect(add_defaults, [10, 20.0], 30.0)

def call_add_defaults():
  return add_defaults(10)

def test_call_add_defaults():
  expect(call_add_defaults, [], 12)

def call_add_defaults_with_names():
  return add_defaults(y = 10, x = 20)

def test_call_defaults_with_names():
  expect(call_add_defaults_with_names, [], 30)

def sub(x,y):
  return x - y

def call_pos_with_names():
  return sub(y = 10, x = 20)

def test_call_pos_with_names():
  expect(call_pos_with_names, [], 10)
  
def tuple_default(x = (1,2)):
  return x[0] + x[1]

def test_tuple_default():
  expect(tuple_default, [], 3)
  expect(tuple_default, [(1,3)], 4)
  
if __name__ == '__main__':
  run_local_tests()