import numpy as np
import parakeet
from parakeet import testing_helpers 

int_vec = 300 + np.arange(300, dtype=int)
float_vec = int_vec.astype(float)
bool_vec = float_vec < np.mean(float_vec)

a = np.arange(500, dtype=float).reshape(50,10)
b = np.arange(100,600).reshape(50,10)

def my_sum(xs):
  return parakeet.reduce(parakeet.add, xs, init=0)

def test_int_sum():
  testing_helpers.expect(my_sum, [int_vec], np.sum(int_vec))

def test_float_sum():
  testing_helpers.expect(my_sum, [float_vec], np.sum(float_vec))

def test_bool_sum():
  testing_helpers.expect(my_sum, [bool_vec], np.sum(bool_vec))

def sqr_dist(y, x):
  return sum((x-y)*(x-y))

def test_sqr_dist():
  z = a[0]
  def run_sqr_dist(c):
    return sqr_dist(z, c)
  par_rslt = parakeet.each(run_sqr_dist, a)
  py_rslt = np.array(map(run_sqr_dist, a))
  assert testing_helpers.eq(par_rslt, py_rslt), \
      "Expected %s but got %s" % (py_rslt, par_rslt)

def reduce_2d(Ys):
  init = parakeet.zeros_like(Ys[0])
  return parakeet.reduce(parakeet.add, Ys, init = init, axis = 0)

def test_2d_reduce():
  par_rslt = reduce_2d(a)
  np_rslt = np.sum(a, 0)
  assert testing_helpers.eq(par_rslt, np_rslt), \
    "Expected %s but got %s" % (np_rslt, par_rslt)

def avg_along_axis_0(Xs):
  assign = np.array([0,0,1,0,1,0,1,0,1,1])
  Ys = Xs[assign == 1]

  def zero(x):
    return 0.0
  zeros = parakeet.each(zero, Xs[0])
  s = reduce(parakeet.add, Ys, init=zeros)
  def d(s):
    return s / Ys.shape[0]
  return parakeet.each(d, s)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
