import numpy as np
import parakeet
import testing_helpers
from testing_helpers import eq 

@parakeet.jit
def range1(n):
  return range(n)

def test_range1():
  assert eq(range1(10), np.arange(10))

@parakeet.jit
def range2(start, stop):
  return range(start, stop)

def test_range2():
  assert eq(range2(100,111), np.arange(100,111)) 

@parakeet.jit
def range3(start, stop, step):
  return range(start, stop, step)

def test_range3():
  assert eq(range3(10,20,2), np.arange(10,20,2))

def test_big_step():
  assert eq(range3(10, 20, 100), np.arange(10,20,100))

if __name__ == '__main__':
  testing_helpers.run_local_tests()
