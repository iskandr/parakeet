"""Helpers for testing compiled functions against NumPy / pure Python."""

from __future__ import annotations

import os

import numpy as np

from .api import available_backends, jit
from .config import config

__all__ = [
    "assert_eq_arrays", "check", "eq", "expect", "expect_allpairs", "expect_each", "expect_eq",
    "run_local_tests", "same", "test_backends",
]  # fmt: skip

_backends: list[str] | None = None


def test_backends() -> list[str]:
  """Backends exercised by :func:`expect` and :func:`check`: the CPU ones that
  work on this machine, or the comma separated list in ``$PARAKEET_TEST_BACKENDS``
  (e.g. ``cuda`` to push a whole test suite through a GPU)."""
  global _backends
  if _backends is None:
    requested = [b.strip() for b in os.environ.get("PARAKEET_TEST_BACKENDS", "").split(",") if b.strip()]
    avail = available_backends()
    _backends = requested or [b for b in ("c", "threads", "gcd", "openmp") if avail.get(b) == "ok"]
  return _backends


test_backends.__test__ = False  # type: ignore[attr-defined]  (not a pytest test)


def eq(x, y, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
  """Structural, tolerance-based equality for scalars, arrays and tuples."""
  if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
    return len(x) == len(y) and all(eq(a, b, rtol, atol) for a, b in zip(x, y))
  if x is None or y is None:
    return x is y
  xa, ya = np.asarray(x), np.asarray(y)
  if xa.shape != ya.shape:
    return False
  if xa.dtype.kind in "biuf" and ya.dtype.kind in "biuf":
    return bool(np.allclose(xa, ya, rtol=rtol, atol=atol, equal_nan=True))
  return bool(np.all(xa == ya))


def expect_eq(actual, expected, label: str = "") -> None:
  assert eq(actual, expected), f"{label + ': ' if label else ''}expected {expected!r} but got {actual!r}"


def _copy(x):
  if isinstance(x, np.ndarray):
    return x.copy()
  if isinstance(x, tuple):
    return tuple(_copy(v) for v in x)
  return x


def expect(fn, args, expected, msg: str | None = None, valid_types=None, backends=None) -> None:
  """Assert that ``fn(*args)`` compiled by Parakeet equals ``expected`` on
  every available CPU backend."""
  rtol = 1e-5
  if hasattr(expected, "dtype") and expected.dtype == "float16":
    # Parakeet has no half floats: it computes in float32, so only compare
    # to the precision the float16 reference actually has
    expected = expected.astype("float32")
    rtol = 5e-3
  compiled = fn if isinstance(fn, jit) else jit(fn)
  for backend in backends or test_backends():
    result = compiled(*[_copy(a) for a in args], _backend=backend)
    label = f"backend={backend}, inputs={', '.join(repr(a) for a in args)}"
    if msg is not None:
      label += f": {msg}"
    assert eq(result, expected, rtol=rtol, atol=rtol * 1e-3), f"{label}: expected {expected!r} but got {result!r}"
    if valid_types is not None:
      if not isinstance(valid_types, (tuple, list)):
        valid_types = [valid_types]
      assert type(result) in valid_types, f"expected result type in {valid_types} but got {type(result)}"


def same(actual, expected, rtol: float = 1e-6) -> str | None:
  """Strict comparison: values, shapes *and* types (Python vs NumPy scalar,
  dtype).  Returns a description of the first difference, or None."""
  if isinstance(expected, tuple):
    if not isinstance(actual, tuple) or len(actual) != len(expected):
      return f"expected a tuple of length {len(expected)}, got {actual!r}"
    for i, (a, e) in enumerate(zip(actual, expected)):
      diff = same(a, e, rtol)
      if diff:
        return f"element {i}: {diff}"
    return None
  if expected is None:
    return None if actual is None else f"expected None, got {actual!r}"
  if isinstance(expected, np.ndarray):
    if not isinstance(actual, np.ndarray):
      return f"expected an array, got {type(actual).__name__} {actual!r}"
    if actual.dtype != expected.dtype:
      return f"expected dtype {expected.dtype}, got {actual.dtype}"
    if actual.shape != expected.shape:
      return f"expected shape {expected.shape}, got {actual.shape}"
    if expected.dtype.kind == "f":
      ok = np.allclose(
        actual, expected, rtol=rtol * (100 if expected.dtype == np.float32 else 1), atol=1e-12, equal_nan=True
      )
    else:
      ok = np.array_equal(actual, expected)
    return None if ok else f"expected\n{expected!r}\ngot\n{actual!r}"
  if isinstance(expected, np.generic):
    # compare dtypes, not classes: np.intc and np.int32 are distinct classes
    # with the same dtype on some platforms
    if not isinstance(actual, np.generic) or actual.dtype != expected.dtype:
      return f"expected {type(expected).__name__} {expected!r}, got {type(actual).__name__} {actual!r}"
  elif type(actual) is not type(expected):
    return f"expected {type(expected).__name__} {expected!r}, got {type(actual).__name__} {actual!r}"
  if isinstance(expected, (float, np.floating)):
    scale = 100 if isinstance(expected, np.float32) else 1
    ok = bool(np.isclose(actual, expected, rtol=rtol * scale, atol=1e-12, equal_nan=True))
  else:
    ok = actual == expected
  return None if ok else f"expected {expected!r}, got {actual!r}"


def check(fn, *args, backends=None, rtol: float = 1e-6, **kwargs):
  """Run ``fn`` in Python and compiled on every backend; results (including
  their types) and any in-place changes to array arguments must agree."""
  py_fn = fn.fn if isinstance(fn, jit) else fn
  compiled = fn if isinstance(fn, jit) else jit(fn)
  ref_args = [_copy(a) for a in args]
  with np.errstate(all="ignore"):
    expected = py_fn(*ref_args, **kwargs)
  result = None
  for backend in backends or test_backends():
    run_args = [_copy(a) for a in args]
    result = compiled(*run_args, _backend=backend, **kwargs)
    diff = same(result, expected, rtol)
    assert diff is None, f"[{backend}] {py_fn.__name__}: {diff}"
    for i, (r, e) in enumerate(zip(run_args, ref_args)):
      if isinstance(e, np.ndarray):
        d = same(r, e, rtol)
        assert d is None, f"[{backend}] {py_fn.__name__}: argument {i} after the call: {d}"
  return result


def expect_each(parakeet_fn, python_fn, inputs) -> None:
  for x in inputs:
    expect(parakeet_fn, [x], python_fn(_copy(x)))


def expect_allpairs(parakeet_fn, python_fn, inputs) -> None:
  for x in inputs:
    for y in inputs:
      expect(parakeet_fn, [x, y], python_fn(_copy(x), _copy(y)))


def assert_eq_arrays(numpy_result, parakeet_result, test_name: str | None = None) -> None:
  prefix = f"[{test_name}] " if test_name else ""
  assert np.shape(numpy_result) == np.shape(parakeet_result), (
    f"{prefix}expected shape {np.shape(numpy_result)} but got {np.shape(parakeet_result)}"
  )
  assert eq(numpy_result, parakeet_result), f"{prefix}expected {numpy_result} but got {parakeet_result}"


def run_local_tests(namespace: dict | None = None) -> None:
  """Run every ``test_*`` function of the calling module (handy in scripts)."""
  import inspect

  if namespace is None:
    frame = inspect.currentframe()
    namespace = frame.f_back.f_globals if frame and frame.f_back else {}
  for name, value in sorted(namespace.items()):
    if name.startswith("test_") and callable(value):
      value()
      print(f"{name}: ok")


_ = config
