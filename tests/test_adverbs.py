"""Parakeet's higher-order operators, inside compiled code and called directly."""

import numpy as np
import pytest

import parakeet
from parakeet import jit
from parakeet.testing import check, eq

RNG = np.random.default_rng(7)
X = RNG.random((6, 4))
Y = RNG.random((5, 4))
V = RNG.random(9)


def test_map_from_python():
  assert eq(parakeet.map(lambda v: v * 2, V), V * 2)
  assert eq(parakeet.each(np.sum, X), X.sum(axis=1))
  assert eq(parakeet.map(np.sum, X, axis=1), X.sum(axis=0))
  assert eq(parakeet.map(lambda v: v + 1, X, axis=None), X + 1)
  assert eq(parakeet.map(lambda a, b: a - b, V, V[::-1]), V - V[::-1])
  assert eq(parakeet.map(lambda row, s: row * s, X, 3.0), X * 3)
  assert eq(parakeet.map(lambda row: row[::-1], X), X[:, ::-1])


def test_map_returning_arrays_of_computed_shape():
  def f(X, k):
    def top(row):
      out = np.zeros(k)
      for i in range(k):
        out[i] = row[i] * 2
      return out

    return parakeet.map(top, X)

  r = jit(f)(X, 3)
  assert eq(r, X[:, :3] * 2)


def test_map_with_data_dependent_shapes_must_agree():
  @jit
  def f(X, n):
    return parakeet.map(lambda i: np.zeros(i + 1), np.arange(n))

  with pytest.raises(ValueError, match="different shapes"):
    f(X, 3)
  assert f(X, 1).shape == (1, 1)
  assert f(X, 0).shape == (0, 0)


def test_nested_maps_and_comprehensions():
  def f(X, Y):
    a = parakeet.map(lambda x: parakeet.map(lambda y: np.sum(x * y), Y), X)
    b = np.array([[np.sum(x * y) for y in Y] for x in X])
    c = [[i * j for j in range(3)] for i in range(4)]
    return a, b, np.array(c)

  a, b, c = jit(f)(X, Y)
  assert eq(a, X @ Y.T) and eq(b, X @ Y.T)
  assert eq(c, np.outer(np.arange(4), np.arange(3)))


def test_allpairs():
  assert eq(parakeet.allpairs(lambda x, y: np.sum((x - y) ** 2), X, Y), ((X[:, None] - Y[None]) ** 2).sum(-1))
  assert eq(parakeet.allpairs(lambda a, b: a * b, V, V), np.outer(V, V))
  assert eq(parakeet.allpairs(np.dot, X, Y.T, axis=(0, 1)), X @ Y.T)
  assert eq(
    parakeet.outer_map(lambda a, b, c: a + b + c, V[:2], V[:3], V[:4]),
    V[:2, None, None] + V[None, :3, None] + V[None, None, :4],
  )


def test_reduce():
  assert eq(parakeet.reduce(parakeet.add, V), V.sum())
  assert eq(parakeet.reduce(parakeet.multiply, V, init=2.0), 2 * V.prod())
  assert eq(parakeet.reduce(lambda a, b: a + b, X), X.sum())
  assert eq(parakeet.reduce(parakeet.add, X, axis=0), X.sum(axis=0))
  assert eq(parakeet.reduce(parakeet.maximum, X, axis=1), X.max(axis=1))
  assert eq(parakeet.reduce(lambda acc, row: acc + row * 2, X, init=np.zeros(4), axis=0), 2 * X.sum(axis=0))
  assert eq(parakeet.reduce(lambda a, b: a * 0.5 + b, V), __import__("functools").reduce(lambda a, b: a * 0.5 + b, V))
  assert parakeet.reduce(parakeet.add, np.arange(10)) == 45
  assert parakeet.reduce(parakeet.add, np.array([True, True, False]), init=0) == 2


def test_reduce_empty():
  with pytest.raises(ValueError, match="zero-size"):
    parakeet.reduce(parakeet.add, np.zeros(0))
  assert parakeet.reduce(parakeet.add, np.zeros(0), init=5.0) == 5.0


def test_large_parallel_reduce_is_exactly_reproducible_for_ints():
  big = np.arange(1_000_000, dtype=np.int64)
  assert parakeet.reduce(parakeet.add, big) == big.sum()
  assert parakeet.reduce(parakeet.maximum, big[::-1]) == 999_999
  assert parakeet.reduce(lambda a, b: a ^ b, big) == np.bitwise_xor.reduce(big)


def test_scan():
  assert eq(parakeet.scan(parakeet.add, V), np.cumsum(V))
  assert eq(parakeet.scan(parakeet.multiply, V, init=1.0), np.cumprod(V))
  assert eq(parakeet.scan(parakeet.add, X, axis=0), np.cumsum(X, axis=0))
  assert eq(parakeet.scan(parakeet.maximum, X, axis=1), np.maximum.accumulate(X, axis=1))
  assert eq(parakeet.scan(lambda a, b: a + b, np.arange(5)), np.cumsum(np.arange(5)))
  assert parakeet.scan(parakeet.add, np.zeros(0)).shape == (0,)


def test_imap_and_parfor():
  assert eq(parakeet.imap(lambda i: i * 2.5, 6), np.arange(6) * 2.5)
  assert eq(parakeet.imap(lambda idx: idx[0] * 10 + idx[1], (3, 4)), np.arange(3)[:, None] * 10 + np.arange(4))

  @jit
  def fill(n):
    out = np.zeros((n, 3))

    def body(idx):
      i, j = idx
      out[i, j] = i - j

    parakeet.parfor(body, (n, 3))
    return out

  assert eq(fill(4), np.arange(4)[:, None] - np.arange(3))
  assert eq(fill(5000), np.arange(5000)[:, None] - np.arange(3))


def test_ireduce_and_filter():
  assert parakeet.ireduce(lambda i: i * i, 10) == sum(i * i for i in range(10))
  assert parakeet.ireduce(lambda idx: idx[0] * idx[1], (4, 5), init=0, combine=parakeet.add) == 60
  assert eq(parakeet.filter(lambda v: v > 0.5, V), V[V > 0.5])


def test_builtin_map_and_functools_reduce():
  import functools

  def f(v):
    doubled = map(lambda t: t * 2, v)
    return np.array(doubled), functools.reduce(lambda a, b: a + b, v), functools.reduce(lambda a, b: a + b, v, 10.0)

  a, b, c = jit(f)(V)
  assert eq(a, V * 2) and eq(b, V.sum()) and eq(c, V.sum() + 10)


def test_patch_maps():
  x = np.arange(7.0)
  assert eq(parakeet.pmap1(np.sum, x, 3), np.convolve(x, np.ones(3), mode="same"))
  img = RNG.random((6, 5))
  out = parakeet.pmap2(np.max, img, (3, 3))
  assert out.shape == img.shape and np.all(out >= img)
  trimmed = parakeet.pmap2_trim(np.mean, img, (3, 3))
  assert trimmed.shape == (4, 3)
  assert np.isclose(trimmed[0, 0], img[:3, :3].mean())


def test_adverb_argument_errors():
  with pytest.raises(parakeet.CompileError, match="tuple"):
    parakeet.map(lambda v: (v, v), V)
  with pytest.raises(TypeError, match="map"):
    parakeet.map(lambda v: v)
  with pytest.raises(ValueError, match="different lengths"):
    parakeet.map(lambda a, b: a + b, V, V[:3])
  with pytest.raises(parakeet.CompileError, match="axis 5 is out of bounds"):
    parakeet.map(np.sum, X, axis=5)


def test_fusion_produces_a_single_loop():
  @jit
  def f(x, alpha, beta):
    t = x * alpha
    u = np.tanh(t)
    return u + beta

  ir = f.specialize(V, 0.5, 0.3, _backend="c").ir
  assert ir.count("alloc(") == 1 and ir.count("for ") == 1

  @jit
  def g(x, y):
    return np.sum((x - y) ** 2)

  ir = g.specialize(V, V, _backend="c").ir
  assert "alloc(" not in ir and ir.count("for ") == 1


def test_fusion_can_be_disabled():
  @jit(fusion=False)
  def f(x):
    return (x * 2 + 1).sum()

  assert f.specialize(V, _backend="c").ir.count("alloc(") == 2
  assert np.isclose(f(V), (V * 2 + 1).sum())


def test_python_backend_runs_the_original_function():
  calls = []

  @jit(backend="python")
  def f(x):
    calls.append(1)
    return x + 1

  assert f(1) == 2 and calls == [1]
  check(lambda x: x + 1, 1)


def test_reduce_preserves_value_changing_casts_as_non_associative():
  """A combiner that applies a value-changing cast (like int64 -> bool or narrowing)
  is not associative and must match an ordered Python left fold across all backends."""
  import functools

  def count_nonzero(acc, x):
    return acc + np.bool_(x)

  @jit
  def count_c(x):
    return parakeet.reduce(count_nonzero, x, init=0)

  arr = np.ones(200_000, dtype=np.int64)
  arr[::5] = 0
  expected = functools.reduce(lambda acc, v: acc + bool(v), arr.tolist(), 0)
  with parakeet.options(backend="c"):
    assert int(count_c(arr)) == expected
  with parakeet.options(backend="threads"):
    assert int(count_c(arr)) == expected

