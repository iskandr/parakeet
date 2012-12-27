import matplotlib.pylab as pylab
import numpy as np
import parakeet
from testing_helpers import expect, expect_each, run_local_tests

size = 401
init = np.array(([0] * (size/2)) + [1] + ([0] * (size - size/2 - 1)))

def rule30(extended, i):
  a, b, c = extended[[i-1,i,i+1]]
  if ((a == 1 and b == 0 and c == 0) or
      (a == 0 and b == 1 and c == 1) or
      (a == 0 and b == 1 and c == 0) or
      (a == 0 and b == 0 and c == 1)):
    return 1
  else:
    return 0

use_parakeet = True
def test_rule30():
  output = init.copy()
  cur = init
  zero_array = np.array([0])
  for _ in range(size/2):
    extended = np.concatenate((zero_array, cur, zero_array))
    def run_rule30(i):
      return rule30(extended, i)
    if use_parakeet:
      cur = parakeet.each(run_rule30, np.arange(1,size+1))
    else:
      cur = np.array(map(run_rule30, range(1,size+1)))
    output = np.vstack((output,cur))

  if not use_parakeet:
    pylab.matshow(output,cmap=pylab.cm.gray)
    pylab.show()

if __name__ == '__main__':
  run_local_tests()
