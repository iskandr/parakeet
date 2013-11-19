import numpy as np
from parakeet.testing_helpers import run_local_tests, expect_eq 
import parakeet

def mk_scalar_add(t):
  f, b, (x,y) = parakeet.build_fn([t,t], t)
  b.return_(b.add(x,y))
  return f

def mk_vec_add(array_t):
  elt_t = parakeet.elt_type(array_t)
  add_fn = mk_scalar_add(elt_t)
  fn, builder, inputs = parakeet.build_fn([array_t, array_t, array_t])
  x,y,z = inputs
  n = builder.len(x)
  def loop_body(idx):
    elt_x = builder.index(x, idx)
    elt_y = builder.index(y, idx)
    builder.setidx(z, idx, builder.call(add_fn, (elt_x, elt_y)))
  builder.loop(0, n, loop_body)
  builder.return_(builder.none)
  return fn 


def vec_add(x,y):
  assert x.dtype == y.dtype 
  assert len(x.shape) == len(y.shape) == 1 
  z = np.zeros_like(x)
  array_t = parakeet.typeof(x)
  fn = mk_vec_add(array_t)
  parakeet.run_typed_fn(fn, (x,y,z))
  return z 
  
def test_vec_add(): 
  xs,ys = np.array([1,2,3]), np.array([10,20,30])
  zs = vec_add(xs, ys)
  expected = xs + ys 
  expect_eq(zs, expected)
  

if __name__ == '__main__': 
  run_local_tests()  
