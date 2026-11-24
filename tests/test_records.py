from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import parakeet as pk


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
  assert dsl.show_rewrite(gbdt_best_split, *args).after_scatters == 1
  with pk.options(skip_transforms=("sibling fusion",)):
    expected = gbdt_best_split(*args)
  np.testing.assert_allclose(gbdt_best_split(*args), expected)
  with np.testing.assert_raises(IndexError):
    hist(np.array([15]), np.array([1]), np.array([]))
