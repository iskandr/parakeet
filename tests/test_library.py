"""NumPy / builtin / math library coverage, checked against NumPy itself."""

import math

import numpy as np
import pytest

import parakeet
from parakeet import jit
from parakeet.testing import check

RNG = np.random.default_rng(42)
A = RNG.random((4, 5))
B = RNG.random((4, 5)) + 0.5
V = RNG.random(7)
I = RNG.integers(-5, 6, size=(3, 4))  # noqa: E741


def test_reductions_all_axes():
  def f(x):
    return (
            x.sum(), x.sum(axis=0), x.sum(axis=1), np.sum(x, axis=-1), x.prod(axis=0), x.min(), x.max(axis=1),
            np.mean(x), x.mean(axis=0), x.std(), x.var(axis=1), np.any(x > 0.5), np.all(x > 0.5, axis=0),
            np.argmin(x), np.argmax(x, axis=0), x.argmax(axis=1), np.count_nonzero(x > 0.3), np.count_nonzero(x > 0.3, axis=1),
        )  # fmt: skip

  check(f, A)
  check(f, I)
  check(f, A.astype(np.float32), rtol=1e-6)
  check(f, I.astype(np.int8))
  check(f, I > 0)


def test_reductions_keepdims_and_tuples():
  def f(x):
    return (
      x.sum(axis=0, keepdims=True),
      x.mean(axis=1, keepdims=True),
      x.sum(axis=(0, 1)),
      x.max(axis=(1, 2)),
      x.sum(axis=(0, 2), keepdims=True),
    )

  check(f, RNG.random((2, 3, 4)))


def test_cumulative():
  def f(x):
    return np.cumsum(x), x.cumsum(axis=0), x.cumsum(axis=1), np.cumprod(x, axis=1), np.cumsum(x[0])

  check(f, A)
  check(f, I)


def test_constructors():
  def f(n, m):
    return (
            np.zeros(n), np.ones((n, m)), np.empty((n, m)).shape, np.full((n,), 2.5), np.zeros((n, m), dtype=np.int32),
            np.ones(n, dtype="float32"), np.arange(n), np.arange(1, n), np.arange(0, n, 2), np.arange(n, 0, -2),
            np.arange(0.5, 3.0, 0.25), np.linspace(0, 1, n), np.linspace(1, 3, n, endpoint=False), np.eye(n),
            np.eye(n, m, 1), np.identity(m), np.array([1, 2, 3]), np.array([[1, 2.5], [3, 4]]), np.array((n, m)),
            np.full((2, 2), n), np.arange(n, dtype=np.float32),
        )  # fmt: skip

  check(f, 4, 3)


def test_like_constructors():
  def f(x):
    return (
      np.zeros_like(x),
      np.ones_like(x),
      np.full_like(x, 3),
      np.empty_like(x).shape,
      np.zeros_like(x, dtype=np.float32),
      x.copy(),
      np.array(x),
      np.asarray(x) is not None,
    )

  check(f, I)
  check(f, A)


def test_shape_manipulation():
  def f(x):
    return (
            x.T, x.reshape(5, 4), x.reshape(-1), x.reshape((2, -1)), np.reshape(x, (10, 2)), x.ravel(), x.flatten(),
            np.transpose(x), x.transpose(1, 0), np.swapaxes(x, 0, 1), np.expand_dims(x, 1), np.squeeze(x[:, :1], axis=1),
            np.squeeze(x[:1], axis=0), np.atleast_2d(x[0]), np.flip(x), np.flip(x, 0), np.fliplr(x), x.T.reshape(-1),
            x[::2].ravel(), x[:, ::2].reshape(-1), np.broadcast_to(x[0], (3, 5)), np.moveaxis(x, 0, 1),
        )  # fmt: skip

  check(f, A)


def test_joining():
  def f(x, y, v):
    return (
            np.concatenate((x, y)), np.concatenate([x, y], axis=1), np.concatenate((v, v, v)), np.stack((x, y)),
            np.stack((x, y), axis=2), np.vstack((x, y)), np.hstack((x, y)), np.hstack((v, v)), np.column_stack((v, v)),
            np.array([v, v * 2]), np.concatenate((x, y), axis=None), np.tile(v, 2), np.tile(x, (2, 1)), np.repeat(v, 2),
        )  # fmt: skip

  check(f, A, B, V)


def test_linear_algebra():
  def f(a, b, v):
    return (
            np.dot(v, v), a @ b.T, np.dot(a, v[:5]), v[:4] @ a, np.dot(a, 2.0), np.vdot(v, v), np.inner(v, v),
            np.outer(v, v[:3]), np.trace(a), np.diag(a), np.diag(v), np.diag(a, 1), np.linalg.norm(v),
            np.linalg.norm(a), np.linalg.norm(v, 1), np.linalg.norm(v, np.inf), np.linalg.norm(a, axis=1),
            np.tril(a), np.triu(a, 1), np.cross(v[:3], v[3:6]),
        )  # fmt: skip

  check(f, A, B, V)


def test_matmul_shape_errors():
  @jit
  def f(a, b):
    return a @ b

  with pytest.raises(ValueError, match="aligned"):
    f(np.zeros((2, 3)), np.zeros((2, 3)))


def test_elementwise_functions():
  def f(x, y):
    return (
            np.where(x > y, x, y), np.where(x > 0.5, 1, 0), np.clip(x, 0.2, 0.8), x.clip(0.3), np.round(x * 10),
            np.around(x, 2), np.abs(x - y), np.sign(x - y), np.square(x), np.sqrt(x), np.exp(x), np.log(y),
            np.log1p(x), np.expm1(x), np.sin(x), np.cos(x), np.tan(x), np.arcsin(x), np.arccos(x), np.arctan(x),
            np.arctan2(x, y), np.sinh(x), np.cosh(x), np.tanh(x), np.arcsinh(x), np.arccosh(y + 1), np.arctanh(x * 0.9),
            np.floor(x * 10), np.ceil(x * 10), np.trunc(x * 10 - 5), np.rint(x * 10), np.minimum(x, y), np.maximum(x, y),
            np.fmin(x, y), np.fmax(x, y), np.hypot(x, y), np.power(x, y), x**2, x**0.5, x**3, np.fmod(x, y), np.mod(x, y),
            np.isnan(x), np.isinf(x), np.isfinite(x), np.logical_and(x > 0.5, y > 1), np.logical_or(x > 0.5, y > 1),
            np.logical_not(x > 0.5), np.logical_xor(x > 0.5, y > 1), np.reciprocal(y), np.negative(x), np.exp2(x),
            np.log2(y), np.log10(y), np.cbrt(x), np.deg2rad(x), np.rad2deg(x), np.logaddexp(x, y), np.logaddexp2(x, y),
            np.copysign(x, -y), np.signbit(x - y), np.heaviside(x - 0.5, 0.5), np.float_power(x, 2), np.positive(x),
            np.fabs(x - y),
        )  # fmt: skip

  check(f, A, B, rtol=1e-12)
  check(f, A.astype(np.float32), B.astype(np.float32), rtol=3e-7)


def test_integer_ufuncs():
  def f(x, y):
    return (
            x & y, x | y, x ^ y, ~x, x << 2, x >> 1, np.bitwise_and(x, y), np.left_shift(x, 1), np.abs(x), np.sign(x),
            np.square(x), np.floor_divide(x, 3), np.remainder(x, 3), np.power(x, 2), np.minimum(x, y), -x, +x, x == y,
            x != y, np.sqrt(np.abs(x)), np.isnan(x), np.floor(x), np.where(x > 0, x, -x),
        )  # fmt: skip

  check(f, I, I[::-1].copy())
  check(f, I.astype(np.int16), I[::-1].astype(np.int16))


def test_nan_handling():
  def f(x):
    return (
      x.min(),
      x.max(),
      np.argmax(x),
      np.argmin(x),
      np.maximum(x, 0.0),
      np.fmax(x, 0.0),
      x.sum(),
      np.isnan(x).sum(),
      np.sign(x),
    )

  check(f, np.array([1.0, np.nan, -2.0, np.inf]))
  check(f, np.array([np.inf, -np.inf, 3.0]))


def test_comparison_helpers():
  def f(x, y):
    return (
      np.allclose(x, y),
      np.allclose(x, x + 1e-10),
      np.isclose(x, y),
      np.array_equal(x, y),
      np.array_equal(x, x.copy()),
      np.array_equal(x, x[:2]),
    )

  check(f, A, B)


def test_differences_and_sorting():
  def f(v):
    return (
      np.diff(v),
      np.diff(v, 2),
      np.sort(v),
      np.argsort(v, kind="stable"),
      np.median(v),
      np.searchsorted(np.sort(v), 0.5),
      np.array(sorted(v)),
    )

  check(f, V)
  check(f, np.array([3, 1, 2, 9, 0, 5, 5, 2]))


def test_diff_axes():
  def f(x):
    return np.diff(x, axis=0), np.diff(x, axis=1), np.diff(x > 0.5)

  check(f, A)


def test_nonzero_and_convolve():
  def f(v):
    return np.nonzero(v > 0.5)[0], np.flatnonzero(v > 0.5), np.where(v > 0.5)[0], np.convolve(v, v[:3])

  check(f, V)


def test_histogram():
  def f(v):
    return np.histogram(v, 5)

  counts, edges = check(f, V)
  assert counts.sum() == len(V)


def test_builtins():
  def f(v, t):
    return (
            len(v), len(t), sum(v), sum(t), min(v), max(v), min(t), max(3, 1, 2), abs(-2.5), any(v > 0.5), all(v > 0.5),
            sum(x * x for x in v), max(x - 0.5 for x in v), tuple(t), bool(t[0]), int(v[0] * 10), float(len(v)),
            round(2.567, 1), round(2.5), round(3.5), pow(2, 5), divmod(17, 5), isinstance(v, np.ndarray), isinstance(t, tuple),
            np.array(list(range(4))), np.array([x * 2 for x in range(3)]), np.array([x for x in v if x > 0.5]), 2 in t, 9 in t,
            0.123 in v, len(range(2, 11, 3)), np.array(list(reversed(range(4)))), sum(range(5)), sum((1, 2, 3)),
        )  # fmt: skip

  check(f, V, (1, 2, 3))


def test_builtin_sum_of_matrix_adds_rows():
  def f(x):
    return sum(x), sum(sum(x))

  check(f, A)


def test_math_module():
  def f(x):
    return (
            math.sqrt(x), math.exp(x), math.log(x), math.log(x, 2), math.log10(x), math.log2(x), math.sin(x), math.cos(x),
            math.tan(x), math.atan(x), math.atan2(x, 2.0), math.sinh(x), math.cosh(x), math.tanh(x), math.floor(x * 3),
            math.ceil(x * 3), math.trunc(-x * 3), math.fabs(-x), math.pow(x, 3), math.hypot(x, 4.0), math.erf(x),
            math.erfc(x), math.gamma(x + 1), math.lgamma(x + 1), math.pi * x, math.e, math.isnan(x), math.isinf(x),
            math.copysign(x, -1.0), math.fmod(x, 0.3), math.degrees(x), math.radians(x), math.expm1(x), math.log1p(x),
            math.asinh(x), math.isfinite(x), math.inf > x,
        )  # fmt: skip

  check(f, 0.75, rtol=1e-12)
  check(f, 2)


def test_math_domain_checks():
  @jit
  def f(x):
    return math.floor(x)

  assert f(2.7) == 2 and type(f(2.7)) is int
  with pytest.raises(ValueError, match="NaN or infinity"):
    f(float("nan"))


def test_properties_and_methods():
  def f(x):
    return (
            x.shape, x.ndim, x.size, x.itemsize, x.nbytes, x.T.shape, x.real, x.dtype == np.float64, x.astype(np.int32),
            x.item(0), x[0].tolist() if False else 0, x.copy().fill(2), x.conj(), x.round(1), x.clip(0.2, 0.4), x.dot(x.T),
            x.trace(), x.diagonal(), x.swapaxes(0, 1), x.squeeze().shape, x.nonzero if False else 0,
        )  # fmt: skip

  check(f, A)


def test_dtype_values():
  def f(x, dt):
    y = np.zeros(3, dtype=x.dtype)
    z = np.ones(2, dtype=dt)
    return (
      y,
      z,
      dt(3),
      x.dtype(2.5) if False else 0,
      np.float32(2.5),
      np.int8(100),
      np.uint8(200),
      np.bool_(5),
      int(2.9),
      float(3),
    )

  check(f, np.arange(3, dtype=np.int16), np.float32)
  check(f, np.arange(3.0), np.int64)


def test_unsupported_function_message():
  @jit
  def f(x):
    return np.fft.fft(x)

  with pytest.raises(parakeet.UnsupportedError, match="fft"):
    f(np.zeros(4))

  @jit
  def g(x):
    return np.cumsm(x)

  with pytest.raises(parakeet.CompileError, match="cumsm"):
    g(np.zeros(4))

  @jit
  def h(x):
    return x.cumsm()

  with pytest.raises(parakeet.UnsupportedError, match="cumsum"):
    h(np.zeros(4))


def test_sort_and_argsort_along_an_axis():
  rng = np.random.default_rng(3)
  a = rng.integers(0, 20, (4, 5, 3)).astype(float)  # plenty of ties: the sort has to be stable
  for axis in (-1, 0, 1, 2, None):
    np.testing.assert_array_equal(jit(lambda a, axis: np.sort(a, axis=axis))(a, axis), np.sort(a, axis=axis))
    np.testing.assert_array_equal(
      jit(lambda a, axis: np.argsort(a, axis=axis))(a, axis), np.argsort(a, axis=axis, kind="stable")
    )
  # the default is the last axis, not the flattened array
  np.testing.assert_array_equal(jit(lambda a: np.sort(a))(a), np.sort(a))
  np.testing.assert_array_equal(jit(lambda a: a.argsort())(a[0]), a[0].argsort(kind="stable"))
  np.testing.assert_array_equal(jit(lambda a: sorted(a))(a[0, 0]), sorted(a[0, 0]))


def test_array_dimensions_become_constants_when_needed():
  @jit
  def zeros_like_rows(x):
    return (0,) * x.shape[1]  # the length of a tuple has to be known when compiling

  assert zeros_like_rows(np.zeros((2, 3))) == (0, 0, 0)
  assert zeros_like_rows(np.zeros((7, 3))) == (0, 0, 0)  # the other dimension is free
  assert zeros_like_rows(np.zeros((2, 5))) == (0,) * 5
  assert len(zeros_like_rows.specializations) == 2


def test_lambdas_on_one_line_are_told_apart():
  x = np.arange(4.0)
  f, g = jit(lambda a, k=1: np.minimum(a, k)), jit(lambda a, k=1: np.maximum(a, k))
  np.testing.assert_array_equal(f(x), np.minimum(x, 1))
  np.testing.assert_array_equal(g(x), np.maximum(x, 1))
