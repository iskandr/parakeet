import numpy as np 

import shape_inference
from shape_inference import symbolic_call_shape, scalar, const, array
from shape_inference import input_shape 
import parakeet 
import testing_helpers


def process_expect(e, fn):
  if e[0] == "input":
    arg_pos = e[1]
    arg_name = fn.args.positional[arg_pos]
    arg_type = fn.type_env[arg_name]
    return input_shape(arg_name, arg_type)
  elif e[0] == "increase-rank":
    nested = process_expect(e[1], fn)
    axis = e[2]
    amt = e[3]
    print nested 
    print axis 
    print amt 
    return shape_inference.increase_rank(nested, axis, amt)
  else:
    raise RuntimeError("Unknown command: %s" % e[0])

def expect_shape(python_fn, args_list, expected):
  typed_fn = parakeet.typed_repr(python_fn, args_list)

  if isinstance(expected, tuple):
    expected = process_expect(expected, typed_fn) 
  
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
  expect_shape(unknown_scalar, [True], scalar)
  
def array_literal():
  return [1,2,3]

def test_array_literal():
  expect_shape(array_literal, [], array(3))

vec = np.array([1,2,3])
mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

def ident(x):
  return x  

def test_ident():
  expect_shape(ident, [vec], ("input", 0))
  expect_shape(ident, [mat], ("input", 0))
  
def increase_rank(x):
  return [x]

def test_increase_rank():
  expected =("increase-rank", ("input", 0), 0, 1)
  expect_shape(increase_rank, [vec], expected)
  expect_shape(increase_rank, [mat], expected)
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()