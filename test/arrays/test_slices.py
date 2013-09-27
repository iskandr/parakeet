import numpy as np
from parakeet import jit 
from parakeet.testing_helpers import expect, expect_each, run_local_tests

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

def assign_four_rows(x, y):
  y[0:2,0:5] = x[0:2,0:5]
  y[0:2,5:10] = x[0:2,5:10]
  y[2:4,0:5] = x[2:4,0:5]
  y[2:4,5:10] = x[2:4,5:10]
  return y

def test_assign_four_rows():
  for m in matrices:
    m_expect = m.copy()
    m_input = m.copy()
    m_zeros = np.zeros_like(m_input)
    expect(assign_four_rows, [m_input, m_zeros], m_expect)

def copy_two_rows(x, y):
  # Fill y with the elements of x to try to get an identical copy
  i = 0
  while i < x.shape[1]:
    y[0][i] = x[0][i]
    y[1][i] = x[1][i]
    i = i + 1
  return y

def loop_slice(x, y, z):
  i = 0
  while i < z.shape[0]:
    i_next = i + 2
    z[i:i_next,:] = copy_two_rows(x[i:i_next,:], y)
    i = i_next
  return z

def test_loop_slices():
  m_input = np.arange(40, dtype=np.int64).reshape(4,10)
  m_zeros = np.zeros((2,10), dtype=np.int64)
  m_z = np.zeros_like(m_input)
  m_expect = m_input.copy()
  expect(loop_slice, [m_input, m_zeros, m_z], m_expect)

def lower_right_corner(X):
  m,n = X.shape
  return X[m/2:m, n/2:n]

def test_lower_right_corner():
  expect_each(jit(lower_right_corner), lower_right_corner, matrices)

def multiple_slices(x):
  y = x[2:4,:]
  return y[:,2]

def test_multiple_slices():
  x = np.random.randn(10,10)
  expect_each(jit(multiple_slices), multiple_slices, matrices)
  

if __name__ == '__main__':
  run_local_tests()
