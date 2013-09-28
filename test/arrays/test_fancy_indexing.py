import numpy as np
from parakeet.testing_helpers import expect, run_local_tests

n = 8
vec_int = np.arange(n)
vec_bool = vec_int % 2
vec_float = vec_bool * 2.75
vectors = [vec_int, vec_bool, vec_float]

indices = np.arange(n)[vec_bool]

def idx(x,i):
  return x[i]

def test_1d_by_1d():
  for v in vectors:
    expect(idx, [v, indices], idx(v,indices))

mat_int = np.array([vec_int]*n)
mat_float = np.array([vec_float]*n)
mat_bool = np.array([vec_bool]*n)
matrices = [mat_int, mat_float, mat_bool]

def test_2d_by_idx():
  for m in matrices:
    expect(idx, [m, indices], idx(m, indices))

if __name__ == '__main__':
    run_local_tests()
