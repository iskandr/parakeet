import numpy as np 
import parakeet
import testing_helpers

def use_zip(x,y):
  return zip(x,y)

def test_zip_tuples():
  a = (1,2,3)
  b = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
  testing_helpers.expect(use_zip, [a,b], zip(a,b))

if __name__ == '__main__':
  testing_helpers.run_local_tests() 

