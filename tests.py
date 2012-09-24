import ast_conversion
import parakeet 
import unittest
import interp

def f(x):
  return x + 1


class TestInterp(unittest.TestCase):
  def test_interp(self):
    untyped, _ = ast_conversion.translate_function_value(f)
    print untyped 
    result = interp.eval_fn(untyped, 1)
    self.assert_(result == 2, "Russ fucked up") 


