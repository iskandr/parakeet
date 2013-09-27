import numpy as np
from parakeet.testing_helpers import expect, run_local_tests

shape_1d = 40
ints_1d = np.arange(shape_1d)
floats_1d = np.arange(shape_1d, dtype='float')
bools_1d = ints_1d % 2

vecs = [ints_1d, floats_1d, bools_1d]

shape_2d = (4,10)
matrices = [np.reshape(vec, shape_2d) for vec in vecs]

shape_3d = (4,5,2)
tensors = [np.reshape(mat, shape_3d) for mat in matrices]

def index_1d(x, i):
  return x[i]

def test_index_1d():
  for vec in vecs:
    expect(index_1d, [vec, 20], vec[20])

def index_2d(x, i, j):
  return x[i, j]

def test_index_2d():
  for mat in matrices:
    expect(index_2d, [mat, 2, 5], mat[2,5])

def index_3d(x, i, j, k):
  return x[i, j, k]

def test_index_3d():
  for x in tensors:
    expect(index_3d, [x, 2, 2, 1], x[2,2,1])




if __name__ == '__main__':
  run_local_tests()
