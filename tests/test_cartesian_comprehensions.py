import numpy as np
import pytest

import parakeet as pk
from parakeet.errors import UnsupportedError


@pk.jit
def product(xs, ys):
  return [10 * x + y for x in xs for y in ys]


@pk.jit
def total(xs, ys):
  return sum(x * y for x in xs for y in ys)


def test_rectangular_order_empty_and_reduction():
  for xs, ys in [(np.arange(3), np.arange(5)), (np.arange(0), np.arange(4)), (np.arange(2), np.arange(0))]:
    np.testing.assert_array_equal(product(xs, ys), [10 * x + y for x in xs for y in ys])
    assert total(xs, ys) == sum(x * y for x in xs for y in ys)
  assert "alloc(" not in total.specialize(np.arange(3), np.arange(5), _backend="c").ir


def test_rectangular_ranges_and_dependent_rejection():
  @pk.jit
  def three(n, m):
    return [100 * i + 10 * j + k for i in range(n) for j in range(m) for k in range(2)]

  np.testing.assert_array_equal(
    three(2, 3), [100 * i + 10 * j + k for i in range(2) for j in range(3) for k in range(2)]
  )

  @pk.jit
  def triangular(xs):
    return [y for x in xs for y in range(x)]

  with pytest.raises(UnsupportedError, match="independent"):
    triangular(np.arange(3))


def test_cartesian_map_rejects_mutation_and_observable_inner_setup():
  def write(a, v):
    a[0] = v
    return v

  @pk.jit
  def writes(xs, out):
    return [write(out, x + y) for x in xs for y in xs]

  @pk.jit
  def unsafe(xs, lengths):
    return [x * j for x in xs for j in range(lengths[0])]

  out = np.array([123])
  with pytest.raises(UnsupportedError, match="pure"):
    writes(np.arange(3), out)
  assert out[0] == 123
  with pytest.raises(UnsupportedError, match="hoist"):
    unsafe(np.arange(0), np.arange(0))
