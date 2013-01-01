import numpy as np
import parakeet
import testing_helpers

def add1(xi):
  return xi+1

int_vec = np.arange(80)

def test_add1():
  result = parakeet.each(add1, int_vec)
  expected  = int_vec + 1
  assert testing_helpers.eq(result, expected), \
      "Expected %s, got %s" % (expected, result)

if __name__ == '__main__':
  testing_helpers.run_local_ts()
