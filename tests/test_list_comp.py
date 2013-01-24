import numpy as np 
import testing_helpers 


def identity_comprehension(x):
  return [xi for xi in x]

def test_identity_comprehension():
  x = np.array([1,2,3,4,5])
  testing_helpers.expect(identity_comprehension, [x], x)
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()
  