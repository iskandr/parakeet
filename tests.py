import ast_conversion
import parakeet 
import unittest
import interp
import numpy as np

 
def add1(x):
  return x + 1

def call_add1(x):
  return add1(x)

def call_nested_ident(x):
  def ident(x):
    return x
  return ident(x)

class TestInterp(unittest.TestCase):
  def test_add1(self):
    assert interp.run(add1, 1) == 2 

  def test_call_add1(self):
    assert interp.run(call_add1, 1) == 2 

