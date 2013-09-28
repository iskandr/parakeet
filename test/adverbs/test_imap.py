import numpy as np
import parakeet
from parakeet import jit 
from parakeet.testing_helpers import expect, run_local_tests, eq


@jit 
def fill_with_const(a, k):
  def return_const(idx):
    return k
  return parakeet.imap(return_const, a.shape)
  
def test_fill_with_const():
  shape=(1,2,1,1)
  a = np.empty(shape=shape, dtype='float32')
  k = 3.137
  expected = np.ones(shape = shape, dtype='float32') * k
  expect(fill_with_const, [a, k], expected)


def test_identity():
  x = np.ones((1,1,2,1))
  def get_idx(idx):
    return x[idx]
  y = parakeet.imap(get_idx, x.shape)
  assert eq(x,y)

if __name__ == '__main__':
  run_local_tests()
