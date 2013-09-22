import numpy as np
from parakeet.testing_helpers import expect, run_local_tests

def return_pair():
  return (-1.0, 200)

def test_return_pair():
  expect(return_pair, [], (-1.0, 200))

ints = (np.int32(1), np.int32(2))
mixed = (1.0, 200L)
nested = (ints, mixed)
nested2 = (nested, nested)

def all_tuples(f, unpack_args = True):
  """
  Given a function which should act as the identity, test it on multiple tuples
  """

  for t in [ints, mixed, nested2, nested2]:
    if unpack_args:
      expect(f, t, t)
    else:
      expect(f, [t], t)

def create_tuple(x,y):
  return (x,y)

def test_create_tuple():
  all_tuples(create_tuple)

#def tuple_arg((x,y)):
#  return (x,y)

#def test_tuple_arg():
#  all_tuples(tuple_arg, unpack_args = False)

def tuple_lhs(t):
  x,y = t
  return x,y

def test_tuple_lhs():
  all_tuples(tuple_lhs, unpack_args = False)

def tuple_lhs_sum(t):
  x, y, z = t
  return x + y + z

def test_tuple_lhs_sum():
  tuples = [(True, 0, 1.0), (1, True, 0)]
  for t in tuples:
    expect(tuple_lhs_sum, [t], sum(t))

def tuple_indexing(t):
  return (t[0], t[1])

def test_tuple_indexing():
  all_tuples(tuple_indexing, unpack_args = False)

def or_elts((b1,b2)):
  if b1 or b2:
    return 1
  else:
    return 0

def test_or_elts():
  expect(or_elts, [(True, True)], 1)
  expect(or_elts, [(True, False)], 1)
  expect(or_elts, [(False, True)], 1)
  expect(or_elts, [(False, False)], 0)

if __name__ == '__main__':
  run_local_tests()
