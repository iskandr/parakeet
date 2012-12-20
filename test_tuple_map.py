import numpy as np
import parakeet
import testing_helpers

def add_tuple_el(t, xi):
  return xi+t[0]

int_vec = np.arange(10)

def test_add1():
  result = parakeet.each(add_tuple_el, (1, 2), int_vec)
  expected  = int_vec + 1
  assert testing_helpers.eq(result, expected), \
    "Expected %s, got %s" % (expected, result)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
