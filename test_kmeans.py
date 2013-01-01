import numpy as np
import parakeet

py_reduce = reduce
py_sum = sum

from parakeet import sum
from prims import add
from testing_helpers import eq, run_local_ts

def ident(x):
  return x

def sqr_dist(x, y):
  return sum((x-y)*(x-y))

def argmin((curmin, curminidx), (val, idx)):
  if val < curmin:
    return (val, idx)
  else:
    return (curmin, curminidx)

inf = float("inf")
def py_minidx(C, idxs, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  dists = map(run_sqr_dist, C)
  zipped = zip(dists, idxs)
  return py_reduce(argmin, zipped, (inf,-1))[1]

def py_calc_centroid(X, a, i):
  cur_members = X[a == i]
  return np.mean(cur_members, axis=0)

def calc_centroid(X, a, i):
  cur_members = X[a == i]
  def zero(x):
    return 0.0
  zeros = each(zero, X[0])
  s = reduce(add, cur_members, init=zeros)
  def d(s):
    return s / cur_members.shape[0]
  return each(d, s)

from parakeet import each, reduce
def par_minidx(C, idxs, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  dists = each(run_sqr_dist, C)
  def z(x, y):
    return (x, y)
  zipped = each(z, dists, idxs)
  return reduce(argmin, zipped, init=(inf,-1))[1]

def py_dists(C, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  return np.array(map(run_sqr_dist, C))

def par_dists(C, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  return each(run_sqr_dist, C)

def par_kmeans(X, assign, k):
  idxs = np.arange(k)
  print assign
  def run_calc_centroid(i):
    return calc_centroid(X, assign, i)

  C = parakeet.each(run_calc_centroid, idxs)
  print "Par C:", C
  converged = False
  j = 0
  while not converged and j < 100:
    lastAssign = assign
    def run_minidx(x):
      return par_minidx(C, idxs, x)
    assign = parakeet.each(run_minidx, X)
    converged = np.all(assign == lastAssign)
    C = parakeet.each(run_calc_centroid, idxs)
    j = j + 1
  return C, assign

def kmeans(X, assign, k):
  idxs = np.arange(k)
  print assign
  def run_calc_centroid(i):
    return py_calc_centroid(X, assign, i)

  C = np.array(map(run_calc_centroid, idxs))
  print "Py C:", C
  converged = False
  j = 0
  while not converged and j < 100:
    lastAssign = assign
    def run_py_minidx(x):
      return py_minidx(C, idxs, x)

    assign = np.array(map(run_py_minidx, X))
    converged = np.all(assign == lastAssign)
    C = map(run_calc_centroid, idxs)
    j = j + 1
  return C, assign

def test_kmeans():
  s = 10
  n = 50
  X = np.random.randn(n*s).reshape(n,s)
  init_assign = np.random.randint(3, size=n)
  k = 3
  C, assign = kmeans(X, init_assign, k)
  par_C, par_assign = par_kmeans(X, init_assign, k)
  assert eq(assign, par_assign), "Expected %s but got %s" % (assign, par_assign)

if __name__ == '__main__':
  run_local_ts()
