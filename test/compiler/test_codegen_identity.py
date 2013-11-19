from parakeet.testing_helpers import run_local_tests, expect_eq

import parakeet


def identity_i64(x):
  fn = parakeet.mk_identity_fn(parakeet.Int64)
  print repr(fn) 
  return parakeet.run_typed_fn(fn, (x,))
  
def identity_f64(x):
  fn = parakeet.mk_identity_fn(parakeet.Float64)
  return parakeet.run_typed_fn(fn, (x,))

def test_identity():
  expect_eq(identity_i64(1), 1)
  expect_eq(identity_i64(-1), -1)
  expect_eq(identity_f64(1.0), 1.0)
  expect_eq(identity_f64(-1.0), -1.0)


if __name__ == '__main__':
 
  run_local_tests()  
