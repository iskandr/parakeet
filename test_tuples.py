import numpy as np 
from parakeet import expect

def return_pair():
    return (-1.0, 200)

def test_return_pair():
    expect(return_pair, [], (-1.0, 200))

def create_tuple(x,y):
    return (x,y)

ints = (np.int32(1), np.int32(2))
mixed = (1.0, 200L)
nested = (ints, mixed)
nested2 = (nested, nested)

def test_all_tuples(f, unpack_args = True):
  """
  Given a function which should act 
  as the identity, test it on multiple tuples
  """
  for t in [ints, mixed, nested2, nested2]:
    if unpack_args:
      expect(f, t, t)
    else:
      expect(f, [t], t) 

def test_create_tuple():  
  test_all_tuples(create_tuple)
  
def tuple_bind((x,y)):
  return (x,y)

#def test_tuple_bind():
#  test_all_tuples(tuple_bind, unpack_args = False)


def tuple_indexing(t):
  return (t[0], t[1])

def test_tuple_indexing():
  test_all_tuples(tuple_indexing, unpack_args = False )

if __name__ == '__main__':
    import testing_helpers
    testing_helpers.run_local_tests()
