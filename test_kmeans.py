import numpy as np
import parakeet

from parakeet import each, reduce
from testing_helpers import eq, run_local_tests

def ident(x):
  return x

def sqr_dist(x, y):
  return sum((x-y) * (x-y))

def argmin((curmin, curminidx), (val, idx)):
  if val < curmin:
    return (val, idx)
  else:
    return (curmin, curminidx)

inf = float("inf")
def minidx(C, idxs, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  dists = each(run_sqr_dist, C)
  def z(x, y):
    return (x, y)
  zipped = each(z, dists, idxs)
  mins_and_idxs = reduce(argmin, zipped, init=(inf,-1))
  def select_second(x):
    return x[1]
  return each(select_second, mins_and_idxs)

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

  #par_C = parakeet.each(run_calc_centroid, idxs)
  C = map(run_calc_centroid, idxs)
  #assert eq(C, par_C), \
  #  "Expected %s but got %s (first iter)" % (C, par_C)
  converged = False
  while not converged:
    lastAssign = assign
    def run_minidx(x):
      return minidx(C, idxs, x)
    #par_assign = parakeet.each(run_minidx, X)
    assign = map(run_minidx, X)
    converged = np.all(assign == lastAssign)
    #assert eq(par_assign, assign), \
    #    "Expected %s but got %s" % (assign, par_assign)
    #par_C = parakeet.each(run_calc_centroid, idxs)
    C = map(run_calc_centroid, idxs)
    #assert eq(par_C, C), \
    #  "Expected %s but got %s" % (C, par_C)
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
