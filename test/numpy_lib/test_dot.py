import numpy as np


from parakeet.testing_helpers import run_local_tests, expect

mat_shape = (2,3)
vec_len = np.prod(mat_shape)
int64_vec = np.arange(vec_len)

def get_vector(t):
  if t == 'bool':
    return int64_vec % 2
  else:
    return int64_vec.astype(t)

def run_vv(xtype, ytype):
  x = get_vector(xtype)
  y = get_vector(ytype)
  expect(np.dot, [x,y], np.dot(x,y))


def run_mv(xtype, ytype):
  x = get_vector(xtype)
  y = get_vector(ytype)
  x = x.reshape(mat_shape)
  y = y[:x.shape[1]]
  expect(np.dot, [x,y], np.dot(x,y))


def run_vm(xtype, ytype):
  x = get_vector(xtype)
  y = get_vector(ytype)
  
  y = y.reshape(mat_shape)
  x = x[:y.shape[0]]

  expect(np.dot, [x,y], np.dot(x,y))

def run_mm(xtype, ytype):
  x = get_vector(xtype)
  y = get_vector(ytype)
  x = x.reshape(mat_shape)
  y = y.reshape(mat_shape).T
  expect(np.dot, [x,y], np.dot(x,y))

#
# Vector-Vector
#
def test_dot_vv_i64_i32():
  run_vv('int64', 'int32')

def test_dot_vv_i64_f32():
  run_vv('int64', 'float32')

def test_dot_vv_i64_bool():
  run_vv('int64', 'bool')

def test_dot_vv_f64_f32():
  run_vv('float64', 'float32')

def test_dot_vv_f64_i32():
  run_vv('float64', 'int32')

def test_dot_vv_f64_bool():
  run_vv('float64', 'bool')

def test_dot_vv_i32_i32():
  run_vv('int32', 'int32')

def test_dot_vv_f32_f32():
  run_vv('float32', 'float32')

def test_dot_vv_bool_bool():
  run_vv('bool', 'bool')
#
# Matrix-Matrix
#

def test_dot_mm_i64_i32():
  run_mm('int64', 'int32')

def test_dot_mm_i64_f32():
  run_mm('int64', 'float32')

def test_dot_mm_bool_f32():
  run_mm('bool', 'float32')

#
#  Matrix-Vector 
#


def test_dot_mv_i64_i32():
  run_mv('int64', 'int32')

def test_dot_mv_i64_i32():
  run_mv('int64', 'int32')

#
# Vector-Matrix
#
def test_dot_vm_i64_i32():
  run_vm('int64', 'int32')

def test_dot_vm_bool_float64():
  run_vm('bool', 'float64')

if __name__ == '__main__':
    run_local_tests()
