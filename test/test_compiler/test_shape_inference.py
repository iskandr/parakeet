import numpy as np
import parakeet
from parakeet import shape_inference
from parakeet.shape_inference import call_shape_expr
from parakeet.shape_inference.shape import const, Shape, Var 
from parakeet import testing_helpers

def expect_shape(python_fn, args_list, expected):
  print "[expect_shape]"
  print " -- fn: ", python_fn
  print " -- args: ", args_list
  typed_fn = parakeet.typed_repr(python_fn, args_list)
  print " -- types: ", typed_fn.input_types
  result_shape = call_shape_expr(typed_fn)
  assert result_shape == expected, \
      "Expected shape %s, but got: %s" % (expected, result_shape)

def const_scalar():
  return 1

def test_const_scalar():
  expect_shape(const_scalar, [], const(1))

def merge_scalars(b):
  if b:
    return 1
  else:
    return 2

def test_any_scalar():
  expect_shape(merge_scalars, [True], shape_inference.any_scalar)

def array_literal():
  return [1,2,3]

def test_array_literal():
  expect_shape(array_literal, [], Shape([3]))

vec = np.array([1,2,3])
mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

def ident(x):
  return x

def test_ident_1d():
  expect_shape(ident, [vec], Shape([Var(0)]))

def test_ident_2d():
  expect_shape(ident, [mat], Shape([Var(0), Var(1)]))

def increase_rank(x):
  return [x,x]

def test_increase_rank_1d():
  expect_shape(increase_rank, [1], Shape([2]))

def test_increase_rank_2d():
  expect_shape(increase_rank, [vec], Shape([2, Var(0)]))

def test_increase_rank_3d():
  expect_shape(increase_rank, [mat], Shape([2, Var(0), Var(1)]))

def incr(xi):
  return xi + 1


def simple_map(x):
  return parakeet.map(incr, x)

def test_simple_map_1d():
  expect_shape(simple_map, [vec], Shape([Var(0)]))

def test_simple_map_2d():
  expect_shape(simple_map, [mat], Shape([Var(0), Var(1)]))

def map_increase_rank(x):
  return parakeet.map(increase_rank, x)

def test_map_increase_rank_1d():
  expect_shape(map_increase_rank, [vec], Shape([Var(0), 2]))

def test_map_increase_rank_2d():
  expect_shape(simple_map, [mat], Shape([Var(0), Var(1)]))

if __name__ == '__main__':
  testing_helpers.run_local_tests()
