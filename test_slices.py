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

def copy_four_rows_by_slices(x, y):
  y[0:2,0:5] = x[0:2,0:5]
  y[0:2,5:10] = x[0:2,5:10]
  y[2:4,0:5] = x[2:4,0:5]
  y[2:4,5:10] = x[2:4,5:10]
  return y

def test_copy_four_rows_by_slices():
  for m in matrices:
    m_expect = m.copy()
    m_input = m.copy()
    m_zeros = np.zeros_like(m_input)
    expect(copy_four_rows_by_slices, [m_input, m_zeros], m_expect)

def copy_first_two_rows(x, y):
  # Fill y with the elements of x to try to get an identical copy
  i = 0
  while i < 10:
    y[0][i] = x[0][i]
    y[1][i] = x[1][i]
    i = i + 1
  return y



def copy_with_slice_loop(x, y, z):
  i = 0
  while i < 10:
    i_next = i + 2
    z[i:i_next,:] = copy_first_two_rows(x[i:i_next,:], y[i:i_next,:])
    i = i_next
  return z

def test_copy_with_slice_loop():
  m_input = np.arange(100).reshape(10,10)
  m_zeros = np.zeros_like(m_input)
  m_z = m_zeros.copy()
  m_expect = np.arange(100).reshape(10,10)
  expect(copy_with_slice_loop, [m_input, m_zeros, m_z], m_expect)



if __name__ == '__main__':
  run_local_tests()
