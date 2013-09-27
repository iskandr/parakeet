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

def set_idx_1d(arr,i,val):
  arr[i] = val
  return arr 

def test_set_idx_1d():
  idx = 10
  for vec in vecs:
    vec1, vec2 = vec.copy(), vec.copy()
    val = -vec[idx]
    vec2[idx] = val
    expect(set_idx_1d, [vec1, idx, val], vec2)

def set_idx_2d(arr,i,j,val):
  arr[i, j] = val
  return arr

def test_set_idx_2d():
  i = 2
  j = 2
  for mat in matrices:
    mat1, mat2 = mat.copy(), mat.copy()
    val = -mat[i,j]
    mat2[i,j] = val
    expect(set_idx_2d, [mat1, i, j, val], mat2)

def set_idx_3d(arr, i, j, k, val):
  arr[i, j, k] = val
  return arr

def test_set_idx_3d():
  i = 2
  j = 3
  k = 1
  for x in tensors:
    x1, x2 = x.copy(), x.copy()
    val = -x[i, j, k]
    x2[i, j, k] = val
    expect(set_idx_3d, [x1, i, j, k, val], x2)


if __name__ == '__main__':
  run_local_tests()
