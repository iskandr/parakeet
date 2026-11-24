from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import parakeet as pk
from parakeet.testing import test_backends


class Pair(NamedTuple):
  x: object
  y: object


@dataclass(frozen=True)
class Encoder:
  weight: object


def test_namedtuple_and_fixed_record_tree():
  @pk.jit
  def swap(p):
    return Pair(p.y, p.x)

  @pk.jit
  def tree(params):
    return params, params["encoder"].weight[1:]

  a = np.arange(5.0)
  p = swap(Pair(a, 2))
  assert type(p) is Pair and p.y is a and p.x == 2
  original = {"encoder": Encoder(a), "bias": a}
  restored, view = tree(original)
  assert list(restored) == list(original)
  assert type(restored["encoder"]) is Encoder
  assert restored["encoder"].weight is restored["bias"] is a
  assert np.shares_memory(view, a)


def test_paired_histogram_fusion():
  from examples.algebraic_gallery import gbdt_best_split
  from parakeet import dsl

  @pk.jit
  def hist(i, a, b):
    return pk.scatter_add(13, i, a), pk.scatter_add(13, i, b)

  rng = np.random.default_rng(15)
  for n in (0, 8, 15000):
    i = rng.integers(0, 13, n)
    a, b = rng.normal(size=n), rng.integers(0, 20, n, dtype=np.int32)
    gh, hh = hist(i, a, b)
    np.testing.assert_allclose(gh, np.bincount(i, weights=a, minlength=13))
    np.testing.assert_array_equal(hh, np.bincount(i, weights=b, minlength=13))
  args = i, a.astype("f"), b.astype("f"), 13, np.float32(1)
  report = dsl.show_rewrite(gbdt_best_split, *args)
  assert (report.before_scatters, report.after_scatters) == (2, 1)
  assert "2 updates in one keyed traversal" in report.summary()
  with pk.options(skip_transforms=("sibling fusion",)):
    expected = gbdt_best_split(*args)
  np.testing.assert_allclose(gbdt_best_split(*args), expected)
  with np.testing.assert_raises(IndexError):
    hist(np.array([15]), np.array([1]), np.array([]))


def test_structured_fold_keeps_chunk_order_and_applies_seed_once():
  from parakeet import dsl

  def compose(left, right):
    return Pair(right.x * left.x, right.x * left.y + right.y)

  @pk.jit
  def recurrence(x):
    return dsl.fold(lambda v: Pair(np.int64(-1), v), compose, x, init=Pair(np.int64(2), np.int64(7)))

  for n in (0, 1, 10001):
    x = np.arange(n, dtype=np.int64)
    expected = Pair(2, 7)
    for v in x:
      expected = compose(expected, Pair(-1, v))
    for backend in test_backends():
      got = recurrence(x, _backend=backend)
      assert type(got) is Pair and got == expected


class Moments(NamedTuple):
  count: int
  mean: float
  m2: float


def test_welford_and_leftmost_argmin_folds():
  from parakeet import dsl

  def join(a, b):
    n = a.count + b.count
    delta = b.mean - a.mean
    return Moments(n, a.mean + delta * b.count / n, a.m2 + b.m2 + delta * delta * a.count * b.count / n)

  @pk.jit
  def variance(x):
    seed = Moments(np.int64(0), np.float64(0), np.float64(0))
    s = dsl.fold(lambda v: Moments(np.int64(1), v, np.float64(0)), join, x, seed)
    return s.m2 / s.count

  @pk.jit
  def lowest(rows):
    def merge(left, right):
      return left if left.x <= right.x else right

    seed = Pair(np.int64(2**62), np.int64(-1))
    return dsl.fold(lambda row: Pair(row[0], row[1]), merge, rows, seed).y

  rng = np.random.default_rng(120)
  x = rng.normal(1e6, 1, 24000)
  keys = rng.integers(1, 10, len(x))
  keys[[19, 21000]] = 0
  rows = np.column_stack((keys, np.arange(len(x))))
  for backend in test_backends():
    np.testing.assert_allclose(variance(x, _backend=backend), np.var(x), rtol=1e-7)
    assert lowest(rows, _backend=backend) == np.argmin(keys)
