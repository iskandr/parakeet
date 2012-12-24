import numpy as np
import testing_helpers

from parakeet import reduce, add, each

int_vec = 100 + np.arange(100, dtype=int)
float_vec = int_vec.astype(float)
bool_vec = float_vec < np.mean(float_vec)

a = np.arange(100).reshape(10,10)
b = np.arange(100,200).reshape(10,10)

def sum(xs):
  return reduce(add, xs, init=0)

def test_int_sum():
  testing_helpers.expect(sum, [int_vec], np.sum(int_vec))

def test_float_sum():
  testing_helpers.expect(sum, [float_vec], np.sum(float_vec))

def test_bool_sum():
  testing_helpers.expect(sum, [bool_vec], np.sum(bool_vec))

if __name__ == '__main__':
  testing_helpers.run_local_tests()
