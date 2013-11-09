import numpy as np
import parakeet

from parakeet.testing_helpers import eq, run_local_tests

size = 21
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

plot = False

def test_rule30():
  output = init.copy()
  cur = init
  zero_array = np.array([0])
  indices = np.arange(1,size+1)
  for _ in range(size/2):
    extended = np.concatenate((zero_array, cur, zero_array))

    def run_rule30(i):
      return rule30(extended, i)
    parakeet_iter = parakeet.each(run_rule30, indices)
    cur = np.array(map(run_rule30, indices))
    assert eq(parakeet_iter, cur), \
       "Parakeet result (%s) didn't match Python result(%s)" % \
       (parakeet_iter, cur)
    output = np.vstack((output,cur))

  if plot:
    import matplotlib.pylab as pylab
    pylab.matshow(output,cmap = pylab.cm.gray)
    pylab.show()

if __name__ == '__main__':
  run_local_tests()
