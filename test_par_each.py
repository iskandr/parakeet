
import numpy as np 
import parakeet 
import testing_helpers

def add1(xi):
  return xi+1

int_vec = np.arange(100)

def test_add1():
  result = parakeet.par_each(add1, int_vec)
  assert testing_helpers.eq(result, int_vec + 1), \
    "Unexpected result: " % result 
  
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()