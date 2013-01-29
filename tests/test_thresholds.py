import testing_helpers
import parakeet 

parakeet.config.print_x86 = True
parakeet.config.print_untyped_function = True 
parakeet.config.print_lowered_function = True
parakeet.config.print_optimized_llvm = True
parakeet.config.print_specialized_function = True
parakeet.config.opt_loop_unrolling = False 

def count_thresh_orig(values, thresh):
  n_left = 0
  n_right = 0
  for elt in values:
    n_left = n_left + (elt < thresh)
    n_right = n_right + (elt >= thresh)
  return n_left, n_right

count_thresh = parakeet.jit(count_thresh_orig)

"""
def gini_orig(values, thresh):
  n_left = 0
  n_right = 0
  for elt in values:
    n_left = n_left + (elt < thresh)
    n_right = n_right + (not (elt<thresh))
  total = 1.0 * n_left + n_right
  return 1.0 - (n_left/total) ** 2 - (n_right/total) ** 2

gini = parakeet.jit(gini_orig)
"""
def test_gini():
  import ast 
  import inspect
  p = ast.parse(inspect.getsource(count_thresh_orig))
  print ast.dump(p)
  import numpy as np
  v = np.array([1.2, 1.4, 5.0, 2, 3])
  parakeet_result = count_thresh(v, 2)
  python_result = count_thresh_orig(v,2)
  assert parakeet_result == python_result, \
    "Parakeet %s != Python %s" % (parakeet_result, python_result)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
