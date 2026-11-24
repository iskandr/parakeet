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
