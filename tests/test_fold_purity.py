import numpy as np
import pytest

from parakeet import dsl
from parakeet.errors import UnsupportedError


def test_fold_rejects_merge_allocations_and_scatter_effects():
  def allocate(left, right):
    scratch = np.zeros(4)
    return (left[0] + right[0] + scratch[0],)

  def scatter(left, right):
    scratch = dsl.iscatter(3, np.arange(3), np.ones(3))
    return (left[0] + right[0] + scratch[0],)

  def scatter_lift(v):
    scratch = dsl.iscatter(3, np.arange(3), np.ones(3))
    return (v + scratch[0],)

  def add(left, right):
    return (left[0] + right[0],)

  def wrap(v):
    return (v,)

  for lift, merge in [(wrap, allocate), (wrap, scatter), (scatter_lift, add)]:
    with pytest.raises(UnsupportedError, match="pure scalar"):
      dsl.fold(lift, merge, np.arange(3.0), (0.0,))
