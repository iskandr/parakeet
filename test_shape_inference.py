import numpy as np 

import shape_inference
from shape_inference import symbolic_call_shape, unknown_scalar, const, array
from shape_inference import Shape, Var 
import parakeet 
import testing_helpers


def expect_shape(python_fn, args_list, expected):
  typed_fn = parakeet.typed_repr(python_fn, args_list)

  result_shape = symbolic_call_shape(typed_fn)
  assert result_shape == expected, \
    "Expected shape %s, but got: %s" % (expected, result_shape) 

def const_scalar():
  return 1

def test_const_scalar():
  expect_shape(const_scalar, [], const(1))

def unknown_scalar(b):
  if b:
    return 1
  else:
    return 2
  
def test_unknown_scalar():
  expect_shape(unknown_scalar, [True], unknown_scalar)
  
def array_literal():
  return [1,2,3]

def test_array_literal():
  expect_shape(array_literal, [], array(3))

vec = np.array([1,2,3])
mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

def ident(x):
  return x  

def test_ident():
  expect_shape(ident, [vec], array(Var(0)))
  expect_shape(ident, [mat], array(Var(0), Var(1)))
  
def increase_rank(x):
  return [x]

def test_increase_rank():
  expect_shape(increase_rank, [vec], array(1, Var(0)))
  expect_shape(increase_rank, [mat], array(1, Var(0), Var(1)))
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()