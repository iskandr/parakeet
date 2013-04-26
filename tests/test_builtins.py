from testing_helpers import expect, eq, run_local_tests
from parakeet import jit
import numpy as np

@jit
def min_(a,b):
  return min(a,b)

def test_min_int():
  expect(min_, [-3, 4], -3)

def test_min_float():
  expect(min_, [-3.0, 4.0], -3.0)

@jit
def max_(a,b):
  return max(a,b)

def test_max_int():
  expect(max_, [3,4], 4)

def test_max_float():
  expect(max_, [3.0,4.0], 4.0)

@jit
def sum_(xs):
  return sum(xs)

def test_sum_int():
  x = np.array([1,2,3])
  expect(sum_, [x], sum(x))

def test_sum_float():
  x = np.array([1.0,2.0,3.0])
  expect(sum_, [x], sum(x))

def sum_bool():
  x = np.array([True, False, True])
  expect(sum_, [x], sum(x))

@jit
def range1(stop):
  return range(stop)

def test_range1():
  expect(range1, [10], np.arange(10))

@jit
def range2(start, stop):
  return range(start, stop)

def test_range2():
  expect(range2, [10, 20], np.arange(10,20))

    
@jit
def range3(start, stop, step):
  return range(start, stop, step) 

def test_range3():
  expect(range3, [20,45,3], np.arange(20,45,3))

@jit 
def to_int(x):
  return int(x)

def test_to_int():
  expect(to_int, [1.2], 1)


@jit 
def to_float(x):
  return float(x)

def test_to_float():
  res = to_float(1)
  assert type(res) is np.float64, "Expected type float, got: %s" % (type(res),)
  assert res == 1.0

  res = to_float(True)
  assert res == 1.0

@jit
def to_long(x):
  return long(x)

def test_to_long():
  res = to_long(-1.0)
  assert res == -1.0

if __name__ == '__main__':
  run_local_tests()
