import unittest
import interp
import numpy as np
from parakeet import expect

def always1():
  return 1

def test_always1():
  expect(always1, [], 1)

def add1(x):
  return x + 1

def test_add1():
  expect(add1, [1],  2) 

def call_add1(x):
  return add1(x)

def test_call_add1():
  expect(call_add1, [1],  2) 

def call_nested_ident(x):
  def ident(x):
    return x
  return ident(x)

def test_nested_ident():
  expect(call_nested_ident, [1], 1)
  
global_val = 5 
def use_global(x):
  return x + global_val 

def test_use_global():
  expect(use_global, [3], 8)

def use_if_exp(x):
  return 1 if x < 10 else 2 

def test_if_exp():
  expect(use_if_exp, [9], 1)
  expect(use_if_exp, [10], 2)
  
def simple_branch(x):
  if x < 10:
    return 1
  else: 
    return 2
    
def test_simple_branch():
  expect(simple_branch, [9], 1)
  expect(simple_branch, [10], 2)
  
def simple_merge(x):
  if x == 0:
    y = 1
  else:
    y = x
  return y 


def test_simple_merge():
  expect(simple_merge, [0], 1)
  expect(simple_merge, [2], 2)
 
def count_loop(init, count):
  x = init
  while x < count:
    x = x + 1
  return x
 
def test_count_loop():
  expect(count_loop, [0, 300], 300 )
  expect(count_loop, [0.0, 400], 400.0)
  expect(count_loop, [0.0, 500.0], 500.0)
 
def if_true():
  if True:
    return 1
  else:
    return 2

def test_if_true():
  expect(if_true, [], 1)
 
#def call_sqrt(x):
#  return np.sqrt(x)

#def test_sqrt():
  #result = interp.run(call_sqrt, 100)
  #assert result == 10, "Expected 10, got %s" % result 
  
  
if __name__ == '__main__':
  import testing_helpers
  testing_helpers.run_local_tests()
  