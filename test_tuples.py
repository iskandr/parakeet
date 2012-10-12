import numpy as np 
from parakeet import expect

def return_pair():
    return (-1.0, 200)

def test_return_pair():
    expect(return_pair, [], (-1.0, 200))

def create_tuple(x,y):
    return (x,y)

def test_create_tuple():
  ints = (np.int16(1), np.int32(2))
  expect(create_tuple, ints, ints)
  mixed = (1.0, 200L)
  expect(create_tuple, mixed, mixed)
  nested = (ints, mixed)
  expect(create_tuple, nested, nested)

if __name__ == '__main__':
    import testing_helpers
    testing_helpers.run_local_tests()
