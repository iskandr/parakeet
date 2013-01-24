import numpy as np
import scipy.spatial
import time

import parakeet
from parakeet import allpairs, each
from testing_helpers import eq, run_local_tests

def python_update_assignments(X, centroids):
  dists = scipy.spatial.distance.cdist(X, centroids, 'sqeuclidean')
  return np.argmin(dists, 1)

def python_update_centroids(X, assignments, k):
  d = X.shape[1]
  new_centroids = np.zeros((k,d), dtype=X.dtype)
  for i in xrange(k):
    mask = assignments == i
    count = np.sum(mask)
    assigned_data = X[mask]
    if count > 1:
      new_centroids[i, :] = np.mean(assigned_data)
    elif count == 1:
      new_centroids[i,:] = assigned_data[0]

  return new_centroids

def python_kmeans(X, k, maxiters = 100, initial_assignments = None):
  n = X.shape[0]
  if initial_assignments is None:
    assignments = np.random.randint(0, k, size=n)
  else:
    assignments = initial_assignments
  centroids = python_update_centroids(X, assignments, k)
  for _ in xrange(maxiters):
    old_assignments = assignments

    assignments = python_update_assignments(X, centroids)

    if all(old_assignments == assignments):
      break
    centroids = python_update_centroids(X, assignments, k)

  return centroids

def sqr_dist(x,y):
  return sum((x-y) ** 2)

def parakeet_update_assignments(X, centroids):
  dists = allpairs(sqr_dist, X, centroids)

  return np.argmin(dists, 1)

def mean(X):
  return sum(X) / len(X)

def parakeet_update_centroids(X, assignments, k):
  d = X.shape[1]

  new_centroids = np.zeros((k,d), dtype=X.dtype)
  for i in xrange(k):
    mask = (assignments == i)
    count = np.sum(mask)
    assigned_data = X[mask]
    if count == 1:
      new_centroids[i, :] = assigned_data[0]
    elif count > 1:
      new_centroids[i,:] = parakeet.mean(assigned_data)
  return new_centroids

def parakeet_kmeans(X, k, maxiters = 100, initial_assignments = None):
  n = X.shape[0]
  if initial_assignments is None:
    assignments = np.random.randint(0, k, size = n)
  else:
    assignments = initial_assignments

  centroids = python_update_centroids(X, assignments, k)
  for _ in xrange(maxiters):
    old_assignments = assignments
    assignments = parakeet_update_assignments(X, centroids)
    if all(old_assignments == assignments):
      break
    centroids = python_update_centroids(X, assignments, k)

  return centroids

def test_kmeans():
  n = 200
  d = 4
  X = np.random.randn(n*d).reshape(n,d)
  k = 2
  niters = 10
  assignments = np.random.randint(0, k, size = n)
  parakeet_C = parakeet_kmeans(X, k, niters, assignments)
  python_C = python_kmeans(X, k, niters, assignments)
  assert eq(parakeet_C.shape, python_C.shape), \
      "Got back wrong shape, expect %s but got %s" % \
      (python_C.shape, parakeet_C.shape)
  assert eq(parakeet_C, python_C), \
      "Expected %s but got %s" % (python_C, parakeet_C)

def test_kmeans_perf():
  n = 1060
  d = 200
  X = np.random.randn(n*d).reshape(n,d)
  k = 20
  niters = 20
  assignments = np.random.randint(0, k, size = n)
#
#  start = time.time()
#  _ = python_kmeans(X, k, niters, assignments)
#  python_time = time.time() - start

  # run parakeet once to warm up the compiler
  start = time.time()
  _ = parakeet_kmeans(X, k, niters, assignments)
  parakeet_with_comp = time.time() - start

  start = time.time()
  _ = parakeet_kmeans(X, k, niters, assignments)
  parakeet_time = time.time() - start

  #speedup = python_time / parakeet_time
  print "Parakeet time:", parakeet_with_comp
  print "Parakeet w/out compilation:", parakeet_time
  #print "Python time", python_time

#  assert speedup > 1, \
#      "Parakeet too slow! Python time = %s, Parakeet = %s, %.1fX slowdown " % \
#      (python_time, parakeet_time, 1/speedup)

if __name__ == '__main__':
  run_local_tests()

"""
parakeet.config.call_from_python_in_parallel = False

py_reduce = reduce
py_sum = sum

from parakeet import  sum, reduce, each
from prims import add
from testing_helpers import eq, run_local_tests

def ident(x):
  return x

def sqr_dist(x, y):
  return sum((x-y)*(x-y))

def argmin((curmin, curminidx), (val, idx)):
  if val < curmin:
    return (val, idx)
  else:
    return (curmin, curminidx)

inf = np.inf
def py_minidx(C, idxs, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  dists = map(run_sqr_dist, C)
  zipped = zip(dists, idxs)
  return py_reduce(argmin, zipped, (inf,-1))[1]

def py_calc_centroid(X, a, i):
  return np.mean(X[a == i], axis=0)

def calc_centroid(X, a, i):
  subset = X[a==i]
  return reduce(add, subset) / subset.shape[0]

def parakeet_zip(xs, ys):
  def make_pair(x, y):
    return (x, y)
  return each(make_pair, xs, ys)

def par_minidx(C, idxs, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  dists = each(run_sqr_dist, C)
  zipped = parakeet_zip(dists, idxs)
  return reduce(argmin, zipped, init=(inf,-1))[1]

def py_dists(C, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  return np.array(map(run_sqr_dist, C))

def par_dists(C, x):
  def run_sqr_dist(c):
    return sqr_dist(c, x)
  return each(run_sqr_dist, C)

def par_kmeans(X, assign, k, max_iters = 100):
  centroid_indices = np.arange(k)
  def run_calc_centroid(i):
    return calc_centroid(X, assign, i)
  C = parakeet.each(run_calc_centroid,
                    centroid_indices)
  for iter_num in xrange(max_iters):
    lastAssign = assign
    def run_minidx(i):
      return par_minidx(C, centroid_indices, i)
    assign = parakeet.each(run_minidx, X)
    if np.all(assign == lastAssign):
      break
    C = parakeet.each(run_calc_centroid,
                      centroid_indices)
  return C, assign

def kmeans(X, assign, k):
  idxs = np.arange(k)
  def run_calc_centroid(i):
    return py_calc_centroid(X, assign, i)

  C = np.array(map(run_calc_centroid, idxs))
  converged = False
  j = 0
  while not converged and j < 100:
    lastAssign = assign
    def run_py_minidx(x):
      return py_minidx(C, idxs, x)

    assign = np.array(map(run_py_minidx, X))
    converged = np.all(assign == lastAssign)
    C = np.array(map(run_calc_centroid, idxs))
    j = j + 1
  return C, assign
"""
