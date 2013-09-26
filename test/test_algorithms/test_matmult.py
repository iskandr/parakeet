import numpy as np

 
from parakeet import jit, allpairs 
from parakeet.testing_helpers import run_local_tests, expect

int_mat = np.reshape(np.arange(6), (2,3))
float_mat = np.sqrt(int_mat)
bool_mat = int_mat % 2

matrices = [int_mat, float_mat, bool_mat]

@jit 
def mm_loops(X,Y,Z):
  m,d = X.shape
  n = Y.shape[1]
    
  for i in range(m):
      for j in range(n):
          total = 0
          for k in range(d):
              total += X[i,k] * Y[k,j]
          Z[i,j] = total 
  return Z

def test_matmult_loops():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y.T)
      Z = np.zeros(res.shape, dtype = res.dtype)
      expect(mm_loops, [X,Y.T,Z], res)
      
@jit 
def dot(x,y):
  return sum(x*y)

@jit 
def matmult_allpairs(X,Y):
  return allpairs(dot, X, Y.T)

def test_matmult_adverb():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y.T)
      expect(matmult_allpairs, [X,Y.T], res)

@jit 
def matmult_comprehensions(X,Y):
  return np.array([[dot(x,y) for y in Y.T] for x in X])

def test_matmult_comprehensions():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y.T)
      expect(matmult_comprehensions, [X,Y.T], res)

def test_matmult_numpy():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y.T)
      expect(np.dot, [X,Y.T], res)

  

if __name__ == '__main__':
    run_local_tests()
