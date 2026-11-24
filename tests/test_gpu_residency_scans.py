"""Tests for GPU-resident scans, small GPU reductions, scalar-load sinking,
and extrema-value reuse (``argmax``/``argmin`` + ``x[idx]``)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import parakeet as pk
from examples.algebraic_gallery import gbdt_best_split


@pk.jit
def _argmax_and_val(x):
  i = np.argmax(x)
  return i, x[i]


@pk.jit
def _argmin_and_val(x):
  i = np.argmin(x)
  return i, x[i]


@pk.jit
def _argmax_expr_and_val(a, b):
  score = a * a - np.float32(2.0) * b
  i = np.argmax(score)
  return i, score[i]


@pk.jit
def _argmax_then_mutate(x):
  y = x.copy()
  i = np.argmax(y)
  y[i] = np.float32(-999.0)
  return i, y[i]


@pk.jit
def _norm_cdf(x):
  c = np.cumsum(x)
  return c / c[len(x) - 1]


@pk.jit
def _seeded_scan_norm(x, init):
  c = pk.scan(pk.add, x, init=init)
  return c - c[len(x) - 1]


@pk.jit
def _load_then_mutate_before_map(x):
  y = x.copy()
  first = y[0]
  y[0] = np.float32(100.0)
  return y + first


@pk.jit
def _oob_load_before_empty_map(x, n):
  val = x[len(x)]
  return np.zeros(n, dtype=np.float32) + val


@pk.jit
def _reduce_suite(x):
  return np.sum(x), np.argmax(x), np.argmin(x)


def _cuda_available() -> bool:
  return os.environ.get("PARAKEET_CUDA_SIMULATE") == "1" or pk.available_backends().get("cuda") == "ok"


def _metal_available() -> bool:
  return sys.platform == "darwin" and pk.available_backends().get("metal") == "ok"


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_extrema_value_reuse_cpu(backend):
  with pk.options(backend=backend):
    # Basic & tie-breaking
    x = np.array([1.0, 5.0, 5.0, -2.0, -2.0], dtype=np.float32)
    assert _argmax_and_val(x) == (1, np.float32(5.0))
    assert _argmin_and_val(x) == (3, np.float32(-2.0))

    # NaN propagation (first NaN wins and its value is NaN)
    xn = np.array([1.0, np.nan, 10.0, np.nan], dtype=np.float32)
    i_max, v_max = _argmax_and_val(xn)
    i_min, v_min = _argmin_and_val(xn)
    assert i_max == 1 and np.isnan(v_max)
    assert i_min == 1 and np.isnan(v_min)

    # Expression value reuse (no recomputation)
    a = np.array([1.0, -3.0, 2.0, 0.5], dtype=np.float32)
    b = np.array([0.0, 1.0, -2.0, 0.5], dtype=np.float32)
    score = a * a - np.float32(2.0) * b
    i, v = _argmax_expr_and_val(a, b)
    assert int(i) == int(np.argmax(score))
    np.testing.assert_allclose(v, score[i], rtol=1e-6)

    # Intervening mutation MUST NOT reuse stale best value
    im, vm = _argmax_then_mutate(x)
    assert int(im) == 1
    assert float(vm) == -999.0

    # Empty array still raises ValueError
    with pytest.raises(ValueError):
      _argmax_and_val(np.array([], dtype=np.float32))


def test_gpu_residency_gbdt_and_scans_cuda():
  if not _cuda_available():
    pytest.skip("CUDA backend/simulator unavailable")
  from parakeet.backends.cuda import CudaDevice

  rng = np.random.default_rng(7)
  ids = rng.integers(256, size=4096, dtype=np.int64)
  g = rng.standard_normal(len(ids), dtype=np.float32) + np.where(ids < 71, -0.3, 0.3).astype(np.float32)
  h = rng.uniform(0.1, 1.0, len(ids)).astype(np.float32)
  gl = np.bincount(ids, weights=g, minlength=256).cumsum()
  hl = np.bincount(ids, weights=h, minlength=256).cumsum()
  gain = gl**2 / (hl + 1.0) + (gl[-1] - gl) ** 2 / (hl[-1] - hl + 1.0)
  ref_bin, ref_gain = int(gain.argmax()), float(gain.max())

  with pk.options(backend="cuda", blas=False, device_fallback="error", device_threshold=1, tensor_cores=False):
    dev = CudaDevice.get()

    # 1. GBDT split: zero uploads, zero intermediate downloads (only the 2 returned scalar buffers)
    d_args = pk.to_device((ids, g, h), backend="cuda")
    gbdt_best_split(*d_args, 256, np.float32(1.0))
    c0 = (dev.launches, dev.uploads, dev.downloads)
    b, gn = gbdt_best_split(*d_args, 256, np.float32(1.0))
    c1 = (dev.launches, dev.uploads, dev.downloads)
    launches, uploads, downloads = (y - x for x, y in zip(c0, c1))
    assert int(b) == ref_bin
    np.testing.assert_allclose(float(gn), ref_gain, rtol=1e-4, atol=1e-4)
    assert launches == 4
    assert uploads == 0
    assert downloads == 2

    # 2. Small (n=256) and large (n=3000) array-returning scan + last-element broadcast: 0 uploads, 0 downloads!
    for n, expected_launches in ((256, 2), (3000, 4)):
      x = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
      dx = pk.to_device(x, backend="cuda")
      _norm_cdf(dx)
      c0 = (dev.launches, dev.uploads, dev.downloads)
      out = _norm_cdf(dx)
      c1 = (dev.launches, dev.uploads, dev.downloads)
      dl, du, dd = (y - x for x, y in zip(c0, c1))
      assert dl == expected_launches
      assert du == 0
      assert dd == 0
      ref = np.cumsum(x, dtype=np.float64)
      ref = (ref / ref[-1]).astype(np.float32)
      np.testing.assert_allclose(pk.to_host(out), ref, rtol=1e-4, atol=1e-4)

    # 3. Seeded scan on both small (n=64) and large (n=2500)
    for n in (64, 2500):
      x = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
      dx = pk.to_device(x, backend="cuda")
      out = _seeded_scan_norm(dx, np.float32(5.0))
      ref = np.cumsum(x, dtype=np.float64) + 5.0
      ref = (ref - ref[-1]).astype(np.float32)
      np.testing.assert_allclose(pk.to_host(out), ref, rtol=1e-4, atol=1e-4)

    # 4. Empty scan stays safe
    empty = np.array([], dtype=np.float32)
    assert pk.to_host(pk.jit(np.cumsum)(pk.to_device(empty, backend="cuda"))).shape == (0,)

    # 5. Scalar load sinking soundness: intervening mutation must preserve old value
    v = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    got = pk.to_host(_load_then_mutate_before_map(pk.to_device(v, backend="cuda")))
    np.testing.assert_allclose(got, np.array([103.0, 7.0, 8.0, 9.0], dtype=np.float32))

    # 6. Out-of-bounds scalar load before an empty ParLoop still raises IndexError
    with pytest.raises(IndexError):
      _oob_load_before_empty_map(pk.to_device(v, backend="cuda"), 0)

    # 7. Small and medium reductions (crossing the 512 single-chunk threshold)
    for n in (1, 17, 256, 512, 513, 1500):
      arr = rng.standard_normal(n).astype(np.float32)
      s, amax, amin = _reduce_suite(pk.to_device(arr, backend="cuda"))
      np.testing.assert_allclose(float(s), float(arr.sum(dtype=np.float64)), rtol=1e-4, atol=1e-4)
      assert int(amax) == int(np.argmax(arr))
      assert int(amin) == int(np.argmin(arr))


def test_gpu_residency_gbdt_and_scans_metal():
  if not _metal_available():
    pytest.skip("Metal backend unavailable")
  from parakeet.backends.metal import MetalDevice

  rng = np.random.default_rng(11)
  ids = rng.integers(256, size=4096, dtype=np.int64)
  g = rng.standard_normal(len(ids), dtype=np.float32) + np.where(ids < 71, -0.3, 0.3).astype(np.float32)
  h = rng.uniform(0.1, 1.0, len(ids)).astype(np.float32)
  gl = np.bincount(ids, weights=g, minlength=256).cumsum()
  hl = np.bincount(ids, weights=h, minlength=256).cumsum()
  gain = gl**2 / (hl + 1.0) + (gl[-1] - gl) ** 2 / (hl[-1] - hl + 1.0)
  ref_bin, ref_gain = int(gain.argmax()), float(gain.max())

  with pk.options(backend="metal", blas=False, device_fallback="error", device_threshold=1):
    dev = MetalDevice.get()
    d_args = pk.to_device((ids, g, h), backend="metal")
    gbdt_best_split(*d_args, 256, np.float32(1.0))
    c0 = (dev.launches, dev.uploads, dev.downloads)
    b, gn = gbdt_best_split(*d_args, 256, np.float32(1.0))
    c1 = (dev.launches, dev.uploads, dev.downloads)
    _, uploads, downloads = (y - x for x, y in zip(c0, c1))
    assert int(b) == ref_bin
    np.testing.assert_allclose(float(gn), ref_gain, rtol=1e-4, atol=1e-4)
    assert uploads == 0
    assert downloads == 2

    for n in (256, 3000):
      x = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
      dx = pk.to_device(x, backend="metal")
      _norm_cdf(dx)
      c0 = (dev.launches, dev.uploads, dev.downloads)
      out = _norm_cdf(dx)
      c1 = (dev.launches, dev.uploads, dev.downloads)
      _, du, dd = (y - x for x, y in zip(c0, c1))
      assert du == 0
      assert dd == 0
      ref = np.cumsum(x, dtype=np.float64)
      ref = (ref / ref[-1]).astype(np.float32)
      np.testing.assert_allclose(pk.to_host(out), ref, rtol=1e-4, atol=1e-4)
