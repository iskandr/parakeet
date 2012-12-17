from testing_helpers import expect, run_local_tests
import numpy as np

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

def implicit_slice_first_axis(x,i):
  return x[i]

def test_implicit_slice_first_axis_matrices():
  for m in matrices:
    expect(implicit_slice_first_axis, [m,2], m[2])

def slice_first_axis(x,i):
  return x[i,:]

def test_slice_first_axis_matrices():
  for m in matrices:
    expect(implicit_slice_first_axis, [m, 2], m[2])

def slice_second_axis(x,i):
  return x[:,i]

def test_slice_second_axis_matrices():
  for m in matrices:
    expect(slice_second_axis, [m,2], m[:,2])

def assign_first_axis(x, i, j):
  x[i] = x[j]
  return x

def test_assign_first_axis():
  for m in matrices:
    m_expect = m.copy()
    m_input = m.copy()
    m_expect[1] = m_expect[2]
    expect(assign_first_axis, [m_input, 1, 2], m_expect)

def assign_second_axis(x, i, j):
  x[:, i] = x[:, j]
  return x

def test_assign_second_axis():
  for m in matrices:
    m_expect = m.copy()
    m_input = m.copy()
    m_expect[:,1] = m_expect[:,2]
    expect(assign_second_axis, [m_input, 1, 2], m_expect)

def assign_slices(x, (i, j, k, l), (a,b,c,d)):
  x[i:j, k:l] = x[a:b, c:d]
  return x

#def test_assign_slices():
#  for m in matrices:
#    m_expect = m.copy()
#    m_input = m.copy()
#    (i,j,k,l) = (0,2,0,4)
#    (a,b,c,d) = (1,3,5,9)
#    m_expect[i:j, k:l] = m_expect[a:b, c:d]
#    print "Running test for %s" % m.dtype
#    expect(assign_slices, [m_input, (i,j,k,l), (a,b,c,d)], m_expect)
#    print "OK #1"
#    expect(assign_slices, [m_input, (i,j,k,l), (a,b,c,d)], m_expect)
#    print "OK #2"

def assign_all_slices(x, y):
  y[0:2,0:5] = x[0:2,0:5]
  y[0:2,5:10] = x[0:2,5:10]
  y[2:4,0:5] = x[2:4,0:5]
  y[2:4,5:10] = x[2:4,5:10]
  return y

def test_assign_all_slices():
  for m in matrices:
    m_expect = m.copy()
    m_input = m.copy()
    m_zeros = np.zeros_like(m_input)
    print "input shape:", m_input.shape
    print "Running test for %s" % m.dtype
    expect(assign_all_slices, [m_input, m_zeros], m_expect)

if __name__ == '__main__':
  run_local_tests()
