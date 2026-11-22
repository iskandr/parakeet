import numpy as np
from parakeet import jit, testing_helpers


values = [
  1,
  1.0,
  np.array([True, True, True]),
  # np.array([1,2,3], dtype='int8'),
  np.array([1, 2, 3], dtype="int16"),
  # np.array([1,2,3], dtype='int32'),
  # np.array([1,2,3], dtype='int64'),
  np.array([1, 2, 3], dtype="float32"),
  # np.array([1,2,3], dtype='float64')
]

import pytest


def run(parakeet_fn, python_fn):
  compiled = jit(parakeet_fn)
  for x in values:
    for y in values:
      try:
        expected = python_fn(x, y)
      except TypeError:
        # NumPy rejects this combination (e.g. bool - bool): so must Parakeet
        with pytest.raises(TypeError):
          compiled(x, y)
        continue
      testing_helpers.expect(compiled, [x, y], expected)


def add(x, y):
  return x + y


def test_add():
  run(add, np.add)


def sub(x, y):
  return x - y


def test_sub():
  run(sub, np.subtract)


def mult(x, y):
  return x * y


def test_mult():
  run(mult, np.multiply)


def div(x, y):
  return x / y


def test_div():
  run(div, np.divide)


def mod(x, y):
  return x % y


def test_mod():
  run(mod, np.mod)


if __name__ == "__main__":
  testing_helpers.run_local_tests()
