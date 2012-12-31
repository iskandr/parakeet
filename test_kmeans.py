import numpy as np
import parakeet

from testing_helpers import eq, run_local_tests

def sqr_dist(x, y):
  return sum((x-y) * (x-y))

def minidx(C, x):
  def run_sqr_dist(y):
    return sqr_dist(x, y)
  #return argmin(parakeet.each(run_sqr_dist, C))
  return np.argmin(run_sqr_dist(C))

def calc_centroid(X, a, i):
  cur_members = X[a == i]
  num_members = cur_members.shape[0]
  if num_members == 0:
    return 0.0
  else:
    return sum(cur_members) / num_members

def kmeans(X, assign, k):
  idxs = np.arange(k)

  def run_calc_centroid(i):
    return calc_centroid(X, assign, i)

  par_C = parakeet.each(run_calc_centroid, idxs)
  C = map(run_calc_centroid, idxs)
  assert eq(C, par_C), \
    "Expected %s but got %s (first iter)" % (C, par_C)
  converged = False
  while not converged:
    lastAssign = assign
    def run_minidx(x):
      return minidx(C, x)
    #par_assign = parakeet.each(run_minidx, X)
    assign = map(run_minidx, X)
    converged = np.all(assign == lastAssign)
    #assert eq(par_assign, assign)
    par_C = parakeet.each(run_calc_centroid, idxs)
    C = map(run_calc_centroid, idxs)
    assert eq(par_C, C), \
      "Expected %s but got %s" % (C, par_C)
  return C, assign

def test_kmeans():
  X = np.array([1,2,3,4,5,6,7,8,9], dtype=np.float)
  assign = np.array([1,0,1,1,1,0,0,0,1])
  k = 2
  C, assign = kmeans(X, assign, k)
  print C
  print assign

if __name__ == '__main__':
  run_local_tests()
