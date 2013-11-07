
import numpy as np 

from parakeet import config 
from parakeet.testing_helpers import run_local_tests, expect 

config.print_transform_times = True 

w,h = 2,2
n = w *h 
vec = np.arange(n)
mat = vec.reshape((w,h))

dtypes = ['int8', 'int16', 'int32', 'int64', 
          'uint8', 'uint16', 'uint32', 'uint64', 
          'float32', 'float64', 'bool']

scalars = [True, 1, 1.0]
vecs = []
mats = []
for dtype_name in dtypes:
  scalars.append(np.dtype(dtype_name).type(3))
  vecs.append(vec.astype(dtype_name))
  mats.append(mat.astype(dtype_name))

def test_log1p_scalar():
  for scalar in scalars:
    print scalar, type(scalar)
    expect(np.log1p, [scalar], np.log1p(scalar))

def test_log1p_vec():
  for vec in vecs:
    expect(np.log1p, [vec], np.log1p(vec))

def test_log1p_mat():
  for mat in mats:
    expect(np.log1p, [mat], np.log1p(mat))

if __name__ == "__main__": 
    run_local_tests()
