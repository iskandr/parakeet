import numpy as np

from parakeet import jit 
from parakeet.testing_helpers import run_local_tests, expect

int_mat = np.reshape(np.arange(6), (2,3))
float_mat = np.sqrt(int_mat)
bool_mat = int_mat % 2

matrices = [int_mat, float_mat, bool_mat]
      
@jit 
def dot(x,y):
  return sum(x*y)


@jit 
def matmult_comprehensions(X,Y):
  return np.array([[dot(x,y) for y in Y.T] for x in X])

def test_matmult_comprehensions():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y.T)
      expect(matmult_comprehensions, [X,Y.T], res)
      
if __name__ == "__main__":
  run_local_tests()