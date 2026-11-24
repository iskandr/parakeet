from dataclasses import dataclass
from typing import NamedTuple

import pytest

import parakeet as pk
from parakeet.errors import TypeInferenceError


class Point(NamedTuple):
  x: float
  y: float = 1.5


@dataclass(frozen=True)
class Weight:
  value: float


def test_namedtuple_defaults_and_argument_errors():
  @pk.jit
  def add(x):
    return Point(x).y + Point(y=x, x=x).x

  assert add(2.0) == 3.5

  @pk.jit
  def missing():
    return Point()

  with pytest.raises(TypeInferenceError, match="missing.*x"):
    missing()


def test_loop_carried_record_activity():
  def read_dict(p):
    state = p
    for _ in range(3):
      state = p
    return state["x"] * state["x"]

  def read_record(p):
    state = p
    for _ in range(3):
      state = p
    return state.value * state.value

  assert pk.grad(read_dict)({"x": 3.0}) == {"x": 6.0}
  assert pk.grad(read_record)(Weight(3.0)) == Weight(6.0)


def test_record_diagnostic_distinguishes_schemas():
  @pk.jit
  def pick(flag, a, b):
    if flag:
      return a
    return b

  with pytest.raises(TypeInferenceError, match=r"incompatible types: (Point.* and Weight|Weight.* and Point)"):
    pick(True, Point(0.0), Weight(0.0))
