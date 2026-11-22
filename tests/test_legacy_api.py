"""The names Parakeet 0.x exported are still there and still mean the same thing."""

import numpy as np
import pytest

import parakeet
from parakeet import Float64, Int64, prims, run_python_fn, specialize, typed_repr


def add1(x):
  return x + 1


def test_run_python_fn():
  x = np.arange(10)
  assert np.array_equal(run_python_fn(add1, [x]), x + 1)
  assert np.array_equal(run_python_fn(add1, [x], backend="c"), x + 1)


def test_specialize_and_typed_repr():
  spec = specialize(add1, [np.arange(10.0)])
  assert "float64" in spec.signature
  assert "float64" in typed_repr(add1, [np.arange(10.0)])


def test_prims_are_first_class():
  x = np.arange(1, 6)
  assert parakeet.reduce(prims.add, x) == 15
  assert parakeet.reduce(prims.multiply, x) == 120
  assert prims.add is parakeet.add and prims.maximum is parakeet.maximum


def test_type_objects():
  assert Int64.dtype == np.int64 and Float64.dtype == np.float64
  assert parakeet.ndtypes.from_dtype(np.dtype("float32")) is parakeet.Float32


def test_old_config_names():
  assert parakeet.config.opt_fusion == parakeet.config.fusion
  with parakeet.options(opt_fusion=False):
    assert parakeet.config.fusion is False
    assert np.allclose(run_python_fn(lambda x: np.sqrt(x * x + 1), [np.arange(4.0)]), np.sqrt(np.arange(4.0) ** 2 + 1))
  assert parakeet.config.fusion is True
  assert parakeet.config.print_generated_code == parakeet.config.print_code


def test_openmp_available_is_a_bool():
  assert parakeet.openmp_available in (True, False)


def test_array_property_functions():
  @parakeet.jit
  def f(x):
    return parakeet.rank(x), parakeet.size(x), parakeet.shape(x)

  assert f(np.zeros((3, 4))) == (2, 12, (3, 4))
  assert parakeet.rank(np.zeros((3, 4))) == 2
  assert abs(parakeet.jit(lambda x: parakeet.erf(x))(0.5) - 0.5204998778) < 1e-9


def test_macro_defines_a_staged_function():
  @parakeet.macro
  def twice(el, x):
    return el.binop("add", x, x)

  @parakeet.jit
  def f(x):
    return twice(x) + 1

  np.testing.assert_array_equal(f(np.arange(4.0)), 2 * np.arange(4.0) + 1)
  # like the built-in adverbs, a macro can be called straight from Python
  np.testing.assert_array_equal(twice(np.arange(3)), [0, 2, 4])
  with pytest.raises(TypeError, match="expects a function"):
    parakeet.macro(3)


def test_find_broken_transform_blames_the_right_step(capsys):
  def scaled(x, a):
    return np.tanh(x * a) + 1.0

  x = np.linspace(-1, 1, 50)
  assert parakeet.find_broken_transform(scaled, [x, 0.5]) is None
  assert "Nothing broke" in capsys.readouterr().out
  assert parakeet.find_broken_transform(scaled, [x, 0.5], expected=x, print_functions=False) == "frontend"
