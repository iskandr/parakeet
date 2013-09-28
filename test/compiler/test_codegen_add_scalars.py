import parakeet
from parakeet.testing_helpers import run_local_tests, expect_eq

def mk_adder(t):
  f, b, (x,y) = parakeet.build_fn([t,t], t)
  b.return_(b.add(x,y))
  return f

def double_i64(x):
  fn = mk_adder(parakeet.Int64)
  return parakeet.run_typed_fn(fn, (x,x))
  
def double_f64(x):
  fn = mk_adder(parakeet.Float64)
  return parakeet.run_typed_fn(fn, (x,x))

def test_add():
  expect_eq(double_i64(1), 2)
  expect_eq(double_i64(-1), -2)
  expect_eq(double_f64(1.0), 2.0)
  expect_eq(double_f64(-1.0), -2.0)


if __name__ == '__main__':
  run_local_tests()  
