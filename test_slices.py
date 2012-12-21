from testing_helpers import expect, run_local_tests
import numpy as np

shape_1d = 40
ints_1d = np.arange(shape_1d)
floats_1d = np.arange(shape_1d, dtype='float')
bools_1d = ints_1d % 2

vecs = [ints_1d, floats_1d, bools_1d]

shape_2d = (4,10)
matrices = [np.reshape(vec, shape_2d) for vec in vecs]

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

def assign_slices(x, (i,j,k,l), (a,b,c,d)):
  x[i:j, k:l] = x[a:b, c:d]
  return x

def test_assign_slices():
  for m in matrices:
    m_expect = m.copy()
    m_input = m.copy()
    (i,j,k,l) = (0,2,0,4)
    (a,b,c,d) = (1,3,5,9)
    m_expect[i:j, k:l] = m_expect[a:b, c:d]
    expect(assign_slices, [m_input, (i,j,k,l), (a,b,c,d)], m_expect)
    expect(assign_slices, [m_input, (i,j,k,l), (a,b,c,d)], m_expect)

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
    expect(assign_all_slices, [m_input, m_zeros], m_expect)

def slice_first(x, y):
  y[0:2,:] = x[0:2,:]
  y[2:4,:] = x[2:4,:]
  return y

def test_slice_first():
  for m in matrices:
    m_expect = m.copy()
    m_y = m.copy()
    for i in range(4):
      m_y[i] = m_expect[3-i]
    m_input = m.copy()
    expect(slice_first, [m_input, m_y], m_expect)

if __name__ == '__main__':
  run_local_tests()
