import numpy as np
import parakeet
from parakeet import jit 
from testing_helpers import expect, run_local_tests


@jit 
def fill_with_const(a, k):
  def setidx(idx):
    a[idx] = k 
  parakeet.parfor(a.shape, setidx)
  
def test_fill_with_const():
  shape=(10,20,30,3)
  a = np.empty(shape=shape, dtype='float32')
  k = 3.137
  expected = np.ones(dtype='float32') * k
  expect(fill_with_const, [a, k], expected)

if __name__ == '__main__':
  run_local_tests()