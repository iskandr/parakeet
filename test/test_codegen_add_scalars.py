from treelike.testing_helpers import run_local_tests, expect_eq

import parakeet

def mk_adder(t):
  f, (x,y), b = parakeet.build_fn([t,t], t)
  b.return_(b.add(x,y))
  return f

def double_i64(x):
  fn = mk_adder(parakeet.Int64)
  return loopjit.run(fn, (x,x))
  
def double_f64(x):
  fn = mk_adder(parakeet.Float64)
  return loopjit.run_typed_fn(fn, (x,x))

def test_add():
  expect_eq(double_i64(1), 2)
  expect_eq(double_i64(-1), -2)
  expect_eq(double_f64(1.0), 2.0)
  expect_eq(double_f64(-1.0), -2.0)


if __name__ == '__main__':
  run_local_tests()  
