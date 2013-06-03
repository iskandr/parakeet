import numpy as np
import parakeet
import testing_helpers

xs = [0.1, 0.5, 2.0]

def test_sin():
  testing_helpers.expect_each(parakeet.sin, np.sin, xs)

def test_sinh():
  testing_helpers.expect_each(parakeet.sinh, np.sinh, xs)

def test_cos():
  testing_helpers.expect_each(parakeet.cos, np.cos, xs)

def test_cosh():
  testing_helpers.expect_each(parakeet.cosh, np.cosh, xs)

def test_tan():
  testing_helpers.expect_each(parakeet.tan, np.tan, xs)

def test_tanh():
  testing_helpers.expect_each(parakeet.tanh, np.tanh, xs)

def test_log():
  testing_helpers.expect_each(parakeet.log, np.log, xs)

def test_log10():
  testing_helpers.expect_each(parakeet.log10, np.log10, xs)

def test_exp():
  testing_helpers.expect_each(parakeet.exp, np.exp, xs)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
