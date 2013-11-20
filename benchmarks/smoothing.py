import numpy as np 


def smooth(x, alpha):
  s = x.copy()
  for i in xrange(1, len(x)):
      s[i] = alpha * x[i] + (1-alpha)*s[i-1]
  return s 

n = 10**6
alpha = 0.01
X = np.random.randn(n).astype('float32')

from compare_perf import compare_perf
compare_perf(smooth, [X, alpha])

