import math 
import numpy as np
import parakeet
from parakeet.testing_helpers import run_local_tests, expect_each

xs = [0.1, 0.5, 2.0]

def test_sin():
  expect_each(parakeet.sin, np.sin, xs)
  expect_each(math.sin, math.sin, xs)
  expect_each(np.sin, np.sin, xs)


def test_parakeet_sinh():
  expect_each(parakeet.sinh, np.sinh, xs)
  expect_each(math.sinh, np.sinh, xs)
  expect_each(np.sinh, np.sinh, xs)


def test_cos():
  expect_each(parakeet.cos, np.cos, xs)
  expect_each(math.cos, np.cos, xs)
  expect_each(np.cos, np.cos, xs)

def test_cosh():
  expect_each(parakeet.cosh, np.cosh, xs)
  expect_each(math.cosh, np.cosh, xs)
  expect_each(np.cosh, np.cosh, xs)

def test_tan():
  expect_each(parakeet.tan, np.tan, xs)
  expect_each(math.tan, np.tan, xs)
  expect_each(np.tan, np.tan, xs)

def test_tanh():
  expect_each(parakeet.tanh, np.tanh, xs)
  expect_each(math.tanh, np.tanh, xs)
  expect_each(np.tanh, np.tanh, xs)
  
def test_log():
  expect_each(parakeet.log, np.log, xs)
  expect_each(math.log, np.log, xs)
  expect_each(np.log, np.log, xs)

def test_log10():
  expect_each(parakeet.log10, np.log10, xs)
  expect_each(math.log10, np.log10, xs)
  expect_each(np.log10, np.log10, xs)

def test_exp():
  expect_each(parakeet.exp, np.exp, xs)
  expect_each(math.exp, np.exp, xs)
  expect_each(np.exp, np.exp, xs)
  
if __name__ == '__main__':
  run_local_tests()
