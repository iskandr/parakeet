

import unittest
import interp
import numpy as np
 
def add1(x):
  return x + 1

def test_add1():
  assert interp.run(add1, 1) == 2 

def call_add1(x):
  return add1(x)

def test_call_add1():
  assert interp.run(call_add1, 1) == 2 

def call_nested_ident(x):
  def ident(x):
    return x
  return ident(x)

def test_nested_ident():
  result = interp.run(call_nested_ident, 1)
  assert result == 1, "Expected 1, got %s" % result 

global_val = 5 
def use_global(x):
  return x + global_val 

def test_use_global():
  assert interp.run(use_global, 3) == 8

def use_if_exp(x):
  return 1 if x < 10 else 2 

def test_if_exp():
  assert interp.run(use_if_exp, 9) == 1
  assert interp.run(use_if_exp, 10) == 2
  
def simple_control_flow(x):
  if x < 10:
    return 1
  else: 
    return 2
    
def test_simple_control_flow():
  result1 = interp.run(simple_control_flow, 9)
  assert result1 == 1, "Expected 1, got %s" % result1
  result2 = interp.run(simple_control_flow, 10)
  assert  result2 == 2, "Expected 2, got %s" % result2  


    
