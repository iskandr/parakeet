"""Training a decision tree, laid out the way CudaTree does it on a GPU.

Every feature keeps its own ordering of the samples.  A node of the tree owns
the same contiguous segment of every ordering, so finding its best split is a
scan along each feature's segment (all features in parallel), and splitting
it is a stable partition of each segment (again all features in parallel).
The tree itself grows in a plain loop over a stack of segments.

Everything is one compiled function: the loops over features are ``parfor``,
their bodies are ordinary loops, and the tree is a handful of arrays.
"""

import numpy as np

import parakeet as par


def gini_after_split(left, right, n_left, n_right):
  """Weighted Gini impurity of two sets, given their per-class counts."""
  sq_left = np.sum(left * left)
  sq_right = np.sum(right * right)
  return (n_left - sq_left / n_left) + (n_right - sq_right / n_right)


def scan_feature(j, X, y, order, lo, hi, totals, counts, scores, positions):
  """Best place to cut feature ``j``'s ordering of the segment [lo, hi).

  ``counts[j]`` is this feature's own scratch space for the class counts to
  the left of the cut, so the loop allocates nothing (a GPU kernel cannot)."""
  left = counts[j]
  left[:] = 0.0
  best, where = np.inf, lo
  for i in range(lo, hi - 1):
    s = order[i, j]
    left[y[s]] += 1.0
    # a cut is only possible between two different values
    if X[s, j] < X[order[i + 1, j], j]:
      n_left = i + 1 - lo
      score = gini_after_split(left, totals - left, n_left, hi - lo - n_left)
      if score < best:
        best, where = score, i + 1
  scores[j] = best
  positions[j] = where


def partition_feature(j, order, scratch, lo, hi, goes_left):
  """Stable partition of feature ``j``'s segment: left samples first."""
  n = 0
  for i in range(lo, hi):
    if goes_left[order[i, j]]:
      scratch[lo + n, j] = order[i, j]
      n += 1
  for i in range(lo, hi):
    if not goes_left[order[i, j]]:
      scratch[lo + n, j] = order[i, j]
      n += 1
  order[lo:hi, j] = scratch[lo:hi, j]


@par.jit
def fit(X, y, n_classes, max_depth, min_samples):
  """Grow a classification tree.  Returns its nodes as arrays: the feature and
  threshold tested at each node (feature -1 for a leaf), the index of its left
  child (the right one follows it) and the class it predicts."""
  n, d = X.shape
  order = np.argsort(X, axis=0)
  scratch = np.zeros((n, d), dtype=np.int64)
  goes_left = np.zeros(n, dtype=np.bool_)
  counts = np.zeros((d, n_classes))
  scores = np.zeros(d)
  positions = np.zeros(d, dtype=np.int64)
  max_nodes = 2 * n + 1
  feature = np.full(max_nodes, -1)
  threshold = np.zeros(max_nodes)
  child = np.zeros(max_nodes, dtype=np.int64)
  label = np.zeros(max_nodes, dtype=np.int64)
  # the segments still to be looked at: (node, lo, hi, depth)
  stack = np.zeros((max_nodes, 4), dtype=np.int64)
  stack[0, 2] = n
  top, n_nodes = 1, 1
  while top > 0:
    top -= 1
    node, lo, hi, depth = stack[top, 0], stack[top, 1], stack[top, 2], stack[top, 3]
    totals = np.zeros(n_classes)
    for i in range(lo, hi):
      totals[y[order[i, 0]]] += 1.0
    label[node] = np.argmax(totals)
    if hi - lo < min_samples or depth >= max_depth or np.max(totals) == hi - lo:
      continue
    par.parfor(lambda j: scan_feature(j, X, y, order, lo, hi, totals, counts, scores, positions), d)
    j = np.argmin(scores)
    if scores[j] == np.inf:
      continue  # no feature takes two different values here
    cut = positions[j]
    feature[node] = j
    threshold[node] = 0.5 * (X[order[cut - 1, j], j] + X[order[cut, j], j])
    child[node] = n_nodes
    for i in range(lo, hi):
      goes_left[order[i, j]] = i < cut
    par.parfor(lambda k: partition_feature(k, order, scratch, lo, hi, goes_left), d)
    stack[top, 0], stack[top, 1], stack[top, 2], stack[top, 3] = n_nodes, lo, cut, depth + 1
    stack[top + 1, 0], stack[top + 1, 1], stack[top + 1, 2], stack[top + 1, 3] = n_nodes + 1, cut, hi, depth + 1
    top += 2
    n_nodes += 2
  return feature[:n_nodes], threshold[:n_nodes], child[:n_nodes], label[:n_nodes]


def descend(x, feature, threshold, child, label):
  node = 0
  while feature[node] >= 0:
    node = child[node] if x[feature[node]] <= threshold[node] else child[node] + 1
  return label[node]


@par.jit
def predict(X, feature, threshold, child, label):
  return np.array([descend(x, feature, threshold, child, label) for x in X])


def best_root_split(X, y, n_classes):
  """The first split, found by trying every threshold of every feature."""
  best = (np.inf, -1, 0.0)
  for j in range(X.shape[1]):
    values = np.unique(X[:, j])
    for t in 0.5 * (values[:-1] + values[1:]):
      side = X[:, j] <= t
      left = np.bincount(y[side], minlength=n_classes).astype(float)
      right = np.bincount(y[~side], minlength=n_classes).astype(float)
      score = gini_after_split(left, right, side.sum(), (~side).sum())
      if score < best[0] - 1e-12:
        best = (score, j, t)
  return best[1], best[2]


def blobs(n, centers, rng):
  y = rng.integers(0, len(centers), n)
  return centers[y] + rng.normal(size=(n, centers.shape[1])), y


def main(n=3000, d=12, n_classes=4, seed=0):
  rng = np.random.default_rng(seed)
  centers = rng.normal(size=(n_classes, d)) * 2.0
  X, y = blobs(n, centers, rng)
  tree = fit(X, y, n_classes, 64, 2)
  feature, threshold, child, label = tree
  # grown to the end, a tree reproduces its training labels
  assert np.array_equal(predict(X, *tree), y)
  j, t = best_root_split(X, y, n_classes)
  assert feature[0] == j and np.isclose(threshold[0], t), (feature[0], threshold[0], j, t)
  # a shallow tree still has to generalise
  X_test, y_test = blobs(n, centers, rng)
  shallow = fit(X, y, n_classes, 6, 2)
  accuracy = float(np.mean(predict(X_test, *shallow) == y_test))
  assert accuracy > 0.8, accuracy
  return len(feature), accuracy


if __name__ == "__main__":
  print("{} nodes, test accuracy of a depth-6 tree {:.3f}".format(*main()))
