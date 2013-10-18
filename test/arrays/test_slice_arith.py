import numpy as np 
from parakeet import testing_helpers

def add_2d(x):
  return x[1:-1, 2:] + x[1:-1, :-2] - 4 * x[1:-1, 1:-1]

def test_add_2d():
  x = np.array([[1,2,3,4,5,6],[10,20,30,40,50,60]])
  testing_helpers.expect(add_2d, [x], add_2d(x))

if __name__ == "__main__":
  testing_helpers.run_local_tests()

