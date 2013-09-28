import numpy as np

from parakeet import each
from parakeet.testing_helpers import (run_local_tests, expect, expect_each, eq, expect_allpairs)

ints_1d = np.arange(300, dtype='int')
floats_1d = np.arange(300, dtype='float')

ints_2d = np.reshape(ints_1d, (25,12))
floats_2d = np.reshape(ints_2d, (25,12))

bools_1d = ints_1d % 2 == 0
bools_2d = ints_2d % 2 == 0

vecs = [ints_1d, floats_1d, bools_1d]
matrices = [ints_2d, floats_2d, bools_2d]
all_arrays = vecs + matrices

def add1_scalar(x):
  return x+1

def test_add1_external_map():
  parakeet_result = each(add1_scalar, ints_1d)
  python_result = ints_1d + 1
  assert eq(parakeet_result, python_result), \
         "Python %s != Parakeet %s" % (python_result, parakeet_result)

def add1_map(x_vec):
  return each(add1_scalar, x_vec)

def test_add1_internal_map_vecs():
  expect_each(add1_map, add1_scalar, vecs)

def test_add1_internal_map_matrices():
  expect_each(add1_map, add1_scalar, matrices)

def add(x,y):
  return x + y

def test_implicit_add_vec():
  expect_allpairs(add, np.add, vecs)

def test_implicit_add_mat():
  expect_allpairs(add, np.add, matrices)

def each_add(x,y):
  return each(add, x, y)

def test_explicit_add_vec():
  expect_allpairs(each_add, np.add, vecs)

def test_explicit_add_mat():
  expect_allpairs(each_add, np.add, matrices)

def conditional_div(x,y):
  if y == 0:
    return 0
  else:
    return x / y

def python_conditional_div(x,y):
  result = [xi / yi if yi != 0 else 0 for (xi,yi) in zip(x,y)]
  return np.array(result)

def each_conditional_div(x,y):
  return each(conditional_div, x, y)

def test_conditional_div():
    expect_allpairs(each_conditional_div, python_conditional_div,
                    [ints_1d, floats_1d])

def second_elt(x):
  return x[1]

X = np.array([[1,2,3],[4,5,6]])

def second_of_columns(X):
  return each(second_elt, X, axis=1)

def test_second_of_columns():
  expect(second_of_columns, [X], np.array([4,5,6]))

def second_of_rows(X):
  return each(second_elt, X, axis=0)

def test_second_of_rows():
  expect(second_of_rows, [X], np.array([2,5]))

def nested_each(x):
  def dummy(x):
    def dummy2():
      return x
    return dummy2()
  return each(dummy, x)

def test_nested_each():
  expect(nested_each, [X], X)

def ident(x):
  return x

def each_ident(x):
  return each(ident, x)

def each_each_ident(x):
  return each(each_ident, x)

def test_map_map():
  expect(each_each_ident, [X], X)

if __name__ == '__main__':
  run_local_tests()
