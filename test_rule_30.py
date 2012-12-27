import matplotlib.pylab as pylab
import numpy as np
import parakeet
from testing_helpers import expect, expect_each, run_local_tests

size = 101
init = np.array(([0] * (size/2)) + [1] + ([0] * (size - size/2 - 1)))

def rule30(idx_vec, old):
  a, b, c = old[idx_vec]
  if ((a == 1 and b == 0 and c == 0) or
      (a == 0 and b == 1 and c == 1) or
      (a == 0 and b == 1 and c == 0) or
      (a == 0 and b == 0 and c == 1)):
    return 1
  else:
    return 0

def test_rule30():
  output = init.copy()
  cur = init
  zero_array = np.array([0])
  idx_vecs = np.array([[i-1, i, i+1] for i in range(1,size+1)])
  for _ in range(100):
    extended = np.concatenate((zero_array, cur, zero_array))
    cur = np.array(map(lambda x: rule30(x, extended), idx_vecs))
    output = np.vstack((output,cur))
  print output.shape
  pylab.matshow(output,fignum=100,cmap=pylab.cm.gray)
  pylab.show()

if __name__ == '__main__':
  run_local_tests()
