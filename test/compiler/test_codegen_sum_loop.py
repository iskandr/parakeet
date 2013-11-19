import numpy as np 
from parakeet.testing_helpers import run_local_tests, expect_eq

import parakeet

def mk_sum(elt_t):
  array_t = parakeet.ndtypes.make_array_type(elt_t, 1)
  f, b, (x,) = parakeet.build_fn([array_t], elt_t)
  n = b.len(x)
  total, total_after, merge = b.loop_var('total', b.zero(elt_t))
  def loop_body(idx):
    b.assign(total_after, b.add(total, b.index(x,idx)))
  b.loop(0, n, loop_body, merge = merge)
  b.return_(total)
  return f 

def sum_i64(x):
  fn = mk_sum(parakeet.Int64)
  print fn 
  return parakeet.run_typed_fn(fn, (x,))
  
def sum_f64(x):
  fn = mk_sum(parakeet.Float64)
  return parakeet.run_typed_fn(fn, (x,))

def test_sum():
  expect_eq(sum_i64(np.array([1,2,3])), 6)
  expect_eq(sum_f64(np.array([-1.0, 1.0, 2.0])), 2.0)
  

if __name__ == '__main__':
 
  run_local_tests()  
