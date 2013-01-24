import numpy as np
import parakeet
import testing_helpers




def test_col_sum():
  m = 500
  n = 5000
  X = np.random.randn(m,n)
  print "Reduce(Map)"
  testing_helpers.timed_test(parakeet.sum, [X], np.sum, [X,0], 
                             min_speedup = 0.1)

def each_col_sum(X):
  return parakeet.each(parakeet.sum, X)

def test_each_col_sum():
  m = 501
  n = 5000
  X = np.random.random((m,n))
  Xt = X.T
  print "Map(Reduce)"
  # summing rows of Xt is same as summing cols of original 
  testing_helpers.timed_test(each_col_sum, [Xt], np.sum, [Xt,1], 
                             min_speedup = 0.1)


if __name__ == '__main__':
  testing_helpers.run_local_tests()

