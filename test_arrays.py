import numpy as np 

from testing_helpers import expect, expect_each, run_local_tests
def create_const(x):
  return [x,x,x,x]

def test_create_const():
  expect(create_const, [1],  np.array([1,1,1,1]))
  expect(create_const, [1.0], np.array([1.0, 1.0, 1.0, 1.0]))
  expect(create_const, [True], np.array([True, True, True, True]))

def set_val(arr,idx,val):
  arr[idx] = val
  return arr 

def test_set_val():
  expect(set_val, [np.array([0,1,2,3]), 1, 100], np.array([0,100,2,3]))  
  expect(set_val, [np.array([0.0,1,2,3]), 1, 100], np.array([0.0, 100.0, 2.0, 3.0]))

shape_1d = 100
ints_1d = np.arange(shape_1d)
floats_1d = np.arange(shape_1d, dtype='float')
bools_1d = ints_1d % 2

vecs = [ints_1d, floats_1d, bools_1d]

def index_1d(x, i):
  return x[i]

def test_index_1d():
  for vec in vecs:
    expect(index_1d, [vec, 50], vec[50])
    
shape_2d = (10,10)
matrices = [np.reshape(vec, shape_2d) for vec in vecs]

def index_2d(x, i, j):
  return x[i, j]

def test_index_2d():
  for mat in matrices:
    expect(index_2d, [mat, 5, 5], mat[5,5])
    
def index_3d(x, i, j, k):
  return x[i, j, k]  

shape_3d = (2,5,10)
tensors = [np.reshape(mat, shape_3d) for mat in matrices]

def test_index_3d():
  for x in tensors:
    expect(index_3d, [x, 1, 2, 3], x[1,2,3])

if __name__ == '__main__':
  run_local_tests()

