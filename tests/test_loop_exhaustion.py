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
