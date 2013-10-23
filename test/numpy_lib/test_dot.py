import numpy as np


from parakeet.testing_helpers import run_local_tests, expect

mat_shape = (2,3)
vec_len = np.prod(mat_shape)

int_vec = np.arange(vec_len)
float_vec = np.sqrt(int_vec)
bool_vec = int_vec % 2 

vectors = [int_vec, float_vec, bool_vec]
matrices = [np.reshape(vec, mat_shape) for vec in vectors]

def test_dot_vv():
  for x in vectors:
    for y in vectors:
      res = np.dot(x,y)
      expect(np.dot, [x,y], res)

def test_dot_mm():
  for X in matrices:
    for Y in matrices:
      res = np.dot(X, Y.T)
      expect(np.dot, [X,Y.T], res)

def test_dot_mv():
  col = np.array([1,2,3])
  for mat in matrices:
    res = np.dot(mat, col)
    expect(np.dot, [mat, col], res)

def test_dot_vm():
  row = np.array([1,2])
  for mat in matrices:
    res = np.dot(row, mat)
    expect(np.dot, [row, mat], res) 

if __name__ == '__main__':
    run_local_tests()
