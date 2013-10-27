import numpy as np 
from parakeet.testing_helpers import run_local_tests, expect 

def use_zip(x,y):
  return zip(x,y)

def test_zip_tuples():
  a = (1,2,3)
  b = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
  expect(use_zip, [a,b], tuple(zip(a,b)))

# TODO: 
# Figure out some plan for returning 'dtype' containing
# multiple 
#def test_zip_arrays():
#  a = np.array([1,2,3])
#  b = np.array([True,False,False])
#  expect(use_zip, [a,b], zip(a,b))


if __name__ == '__main__':
  run_local_tests() 

