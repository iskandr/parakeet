import numpy as np
import pytest

import parakeet as pk


def test_control_flow():
  @pk.jit
  def chain(i, x):
    return i < 0 < x[i]

  assert not chain(10, np.arange(2))

  @pk.jit
  def find(a, b, stop):
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
      if i == stop:
        break
    else:
      return 9
    return i

  a = np.arange(3)
  assert find(a, a, -1) == 9
  assert find(a[:0], a[:0], -1) == 9
  assert find(a, a[:2], 1) == 1
  with pytest.raises(ValueError):
    find(a, a[:2], -1)

  @pk.jit
  def countdown(n):
    while n > 0:
      n -= 1
      if n == 2:
        break
    else:
      return 9
    return n

  assert countdown(4) == 2
  assert countdown(1) == countdown(0) == 9


def test_loop_else_without_break_and_nested_targets():
  @pk.jit
  def search(xs, n):
    for x in xs:
      for j in range(n):
        if x == j:
          break
      else:
        break
    else:
      return 42
    return x

  assert search((1, 2), 3) == 42
  assert search((1, 5), 3) == 5
  assert search((), 3) == 42

  @pk.jit
  def result(n):
    for i in range(n):
      continue
    else:
      x = n + 2
    return x

  assert result(0) == 2 and result(4) == 6


def test_tuple_zip_can_break_before_exhaustion():
  @pk.jit
  def first(xs, ys):
    for x, y in zip(xs, ys, strict=True):
      break
    else:
      return 0
    return x + y

  assert first((1, 2), (4, 5)) == first((1, 2), (4,)) == 5
