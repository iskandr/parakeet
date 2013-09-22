import numpy as np
from parakeet import each, testing_helpers

def add1(xi):
  return xi + 1

def add11d(x):
  return each(add1, x)

int_vec = np.arange(128, dtype=np.float).reshape(16,8)

def test_add1():
  result = each(add11d, int_vec)
  expected = int_vec + 1
  assert testing_helpers.eq(result, expected), \
      "Expected %s, got %s" % (expected, result)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
