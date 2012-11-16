import testing_helpers
import numpy as np 
import adverb_semantics

interp = adverb_semantics.AdverbSemantics()

vec = np.array([1,4,9,16])
mat = np.array([vec, vec+100, vec+200])

expected_sqrt_vec = np.sqrt(vec)
expected_sqrt_mat = np.sqrt(mat)

def test_map():
  sqrt_vec = interp.eval_map(np.sqrt, values=[vec], axis=0)
  assert testing_helpers.eq(sqrt_vec, expected_sqrt_vec), \
    "Expected %s from Map but got %s" % (expected_sqrt_vec, sqrt_vec)
  
  sqrt_mat = interp.eval_map(np.sqrt, values=[mat], axis=0)
  assert testing_helpers.eq(sqrt_mat, expected_sqrt_mat), \
    "Expected %s from Map but got %s" % (expected_sqrt_mat, sqrt_mat)

expected_sum_vec = np.sum(vec)
expected_sum_mat = np.sum(mat, axis=0)

def test_reduce():
  print "Testing vector reduction..."
  vec_sum = interp.eval_reduce(
    map_fn = interp.identity_function, 
    combine = np.add,  
    init = 0, 
    values = [vec], 
    axis = 0) 

  assert testing_helpers.eq(vec_sum, expected_sum_vec), \
    "Expected %s from Reduce but got %s" % (expected_sum_vec, vec_sum)

  print "Testing matrix reduction..."
  mat_sum = interp.eval_reduce(
    map_fn = interp.identity_function, 
    combine = np.add,  
    init = 0, 
    values = [mat], 
    axis = 0) 

  assert testing_helpers.eq(mat_sum, expected_sum_mat), \
    "Expected %s from Reduce but got %s" % (expected_sum_mat, mat_sum)

expected_cumsum_vec = np.cumsum(vec)
expected_cumsum_mat = np.cumsum(mat, axis=1)

def test_scan():
  print "Testing vector scan..."
  vec_prefixes = interp.eval_scan(
    map_fn = interp.identity_function,
    combine = np.add,
    emit = interp.identity_function, 
    init = 0, 
    values = [vec], 
    axis = 0)
  
  assert testing_helpers.eq(vec_prefixes, expected_cumsum_vec), \
    "Expected %s from Scan but got %s" % (expected_cumsum_vec, vec_prefixes)

  print "Testing matrix scan..."
  mat_prefixes = interp.eval_scan(
    map_fn = interp.identity_function,
    combine = np.add,
    emit = interp.identity_function, 
    init = 0, 
    values = [mat], 
    axis = 0)

  assert testing_helpers.eq(mat_prefixes, expected_cumsum_mat), \
    "Expected %s from Scan but got %s" % (expected_cumsum_mat, mat_prefixes)

  
  

if __name__ == '__main__':
  testing_helpers.run_local_tests()
