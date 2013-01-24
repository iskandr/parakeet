import numpy as np
import testing_helpers
import parakeet 

from parakeet.interp import adverb_evaluator as interp

vec = np.array([1,4,9,16])
mat = np.array([vec, vec+100, vec+200, vec + 300])

expected_sqrt_vec = np.sqrt(vec)
expected_sqrt_mat = np.sqrt(mat)

def test_map_1d():
  sqrt_vec = interp.eval_map(np.sqrt, values=[vec], axis=0)
  assert testing_helpers.eq(sqrt_vec, expected_sqrt_vec), \
      "Expected %s from Map but got %s" % (expected_sqrt_vec, sqrt_vec)

def test_map_2d():
  sqrt_mat = interp.eval_map(np.sqrt, values=[mat], axis=0)
  assert testing_helpers.eq(sqrt_mat, expected_sqrt_mat), \
      "Expected %s from Map but got %s" % (expected_sqrt_mat, sqrt_mat)

"""
def sqrt_with_output(x, output):
  output[:] = np.sqrt(x)

def test_map_2d_output():
  out = np.zeros_like(expected_sqrt_mat)
  interp.eval_map(sqrt_with_output, values=[mat], axis=0, output=out)
  assert testing_helpers.eq(out, expected_sqrt_mat), \
      "Expected %s from Map with output param but got %s" % \
      (expected_sqrt_mat, out)
"""

def two_first_elts(x):
    return [x[0], x[0]]

"""
def two_first_elts_with_output(x, out):
    out[:] = two_first_elts(x)

def test_complex_map_2d_output():
  expected = np.array([two_first_elts(x) for x in mat])
  out = np.zeros_like(expected)
  interp.eval_map(two_first_elts_with_output, values=[mat], axis=0, output=out)
  assert testing_helpers.eq(out, expected), \
      "Expected %s from Map with output param but got %s" % \
      (expected, out)
"""

expected_sum_vec = np.sum(vec)

def test_reduce_1d():
  vec_sum = interp.eval_reduce(map_fn = interp.identity_function,
                               combine = np.add,
                               init = 0,
                               values = [vec],
                               axis = 0)

  assert testing_helpers.eq(vec_sum, expected_sum_vec), \
      "Expected %s from Reduce but got %s" % (expected_sum_vec, vec_sum)

expected_sum_mat = np.sum(mat, axis=0)

def test_reduce_2d():
  mat_sum = interp.eval_reduce(map_fn = interp.identity_function,
                               combine = np.add,
                               init = 0,
                               values = [mat],
                               axis = 0)

  assert testing_helpers.eq(mat_sum, expected_sum_mat), \
      "Expected %s from Reduce but got %s" % (expected_sum_mat, mat_sum)

"""
def add_vec_with_output(x, y, out):
    out[:] = x + y

def test_reduce_2d_output():
  output = np.zeros_like(expected_sum_mat)
  interp.eval_reduce(
    map_fn = interp.identity_function,
    combine = add_vec_with_output,
    init = 0,
    values = [mat],
    axis = 0,
    output = output)

  assert testing_helpers.eq(output, expected_sum_mat), \
      "Expected %s from Reduce (with output) but got %s" % \
      (expected_sum_mat, output)
"""
bool_vec = np.array([True, False, True, False, True])

def test_bool_sum():
  vec_sum = interp.eval_reduce(
    map_fn = interp.identity_function,
    combine = (lambda x,y: x + y),
    init = 0,
    values = [bool_vec],
    axis = 0)
  assert vec_sum == np.sum(bool_vec), \
      "Expected %s but got %s" % (np.sum(bool_vec), vec_sum)

expected_cumsum_vec = np.cumsum(vec)

def test_scan_1d():
  vec_prefixes = interp.eval_scan(map_fn=interp.identity_function,
                                  combine=np.add,
                                  emit=interp.identity_function,
                                  init=0,
                                  values=[vec],
                                  axis=0)

  assert testing_helpers.eq(vec_prefixes, expected_cumsum_vec), \
      "Expected %s from Scan but got %s" % (expected_cumsum_vec, vec_prefixes)

expected_cumsum_mat = np.cumsum(mat, axis=0)

def test_scan_2d():
  mat_prefixes = interp.eval_scan(
    map_fn = interp.identity_function,
    combine = np.add,
    emit = interp.identity_function,
    init = 0,
    values = [mat],
    axis = 0)

  assert testing_helpers.eq(mat_prefixes, expected_cumsum_mat), \
      "Expected %s from Scan but got %s" % (expected_cumsum_mat, mat_prefixes)

def test_allpairs():
  times_table = interp.eval_allpairs(np.multiply, vec, vec, 0)
  np_times_table = np.multiply.outer(vec, vec)
  assert testing_helpers.eq(times_table, np_times_table), \
      "Expected %s for AllPairs but got %s" % \
      (np_times_table, times_table)

  inner_products = interp.eval_allpairs(np.dot, mat, mat.T, 0)
  np_inner_products = np.dot(mat, mat)
  assert testing_helpers.eq(inner_products, np_inner_products), \
      "Expected %s for AllPairs but got %s" % \
      (inner_products, np_inner_products)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
