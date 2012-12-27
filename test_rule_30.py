import matplotlib.pylab as pylab
import numpy as np
import parakeet
from testing_helpers import expect, expect_each, run_local_tests, eq

size = 15
init = np.array(([0] * (size/2)) + [1] + ([0] * (size - size/2 - 1)))

def rule30(extended, (a,b,c)):
  if ((a == 1 and b == 0 and c == 0) or
      (a == 0 and b == 1 and c == 1) or
      (a == 0 and b == 1 and c == 0) or
      (a == 0 and b == 0 and c == 1)):
    return 1
  else:
    return 0
plot = False 

def test_rule30():
  output = init.copy()
  cur = init
  zero_array = np.array([0])
  idx_vecs = np.array([[i-1, i, i+1] for i in range(1,size+1)])
  for _ in range(size/2):
    extended = np.concatenate((zero_array, cur, zero_array))
    def run_rule30(idx):
      return rule30(extended, idx)
    parakeet_iter = parakeet.each(run_rule30, idx_vecs)
    python_iter = np.array(map(run_rule30, idx_vecs))
    assert eq(parakeet_iter, python_iter), \
       "Parakeet result (%s) didn't match Python result(%s)" % (parakeet_iter, python_iter)
    output = np.vstack((output,cur))

  if plot:
    pylab.matshow(output,fignum=100,cmap=pylab.cm.gray)
    pylab.show()

if __name__ == '__main__':
  run_local_tests()
