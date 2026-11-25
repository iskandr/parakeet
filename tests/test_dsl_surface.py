import numpy as np
import pytest

from parakeet import dsl


def test_imap_ireduce_parfor_multi_axis_unpacking():
  @dsl.kernel
  def unpacked_2d(a, b):
    m, k = a.shape
    _, n = b.shape
    return dsl.imap(
      lambda i, j: dsl.ireduce(lambda r: a[i, r] + b[r, j], k, combine=dsl.min),
      (m, n),
    )

  @dsl.kernel
  def tuple_compat_2d(m, n):
    return dsl.imap(lambda ij: ij[0] * 10 + ij[1], (m, n))

  @dsl.kernel
  def unpacked_3d(x):
    a, b, c = x.shape
    return dsl.imap(lambda i, j, k: x[i, j, k] + i + j + k, (a, b, c))

  @dsl.kernel
  def ireduce_2d_unpacked(x):
    m, n = x.shape
    return dsl.ireduce(lambda i, j: x[i, j], (m, n), combine=dsl.max)

  @dsl.kernel
  def parfor_2d_unpacked(out, a):
    m, n = a.shape

    def body(i, j):
      out[i, j] = a[i, j] * dsl.f32(2.0)

    dsl.parfor(body, (m, n))
    return out

  rng = np.random.default_rng(7)
  a = rng.normal(size=(7, 5)).astype(np.float32)
  b = rng.normal(size=(5, 6)).astype(np.float32)
  expected_trop = np.min(a[:, :, None] + b[None, :, :], axis=1)

  for backend in ("c", "multicore"):
    np.testing.assert_allclose(unpacked_2d(a, b, _backend=backend), expected_trop, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
      tuple_compat_2d(3, 4, _backend=backend),
      np.arange(3)[:, None] * 10 + np.arange(4)[None, :],
    )
    x3 = rng.normal(size=(2, 3, 4)).astype(np.float32)
    exp3 = x3 + np.arange(2)[:, None, None] + np.arange(3)[None, :, None] + np.arange(4)[None, None, :]
    np.testing.assert_allclose(unpacked_3d(x3, _backend=backend), exp3, rtol=1e-5, atol=1e-5)
    assert np.isclose(ireduce_2d_unpacked(a, _backend=backend), np.max(a))
    out = np.zeros_like(a)
    np.testing.assert_allclose(parfor_2d_unpacked(out, a, _backend=backend), a * 2.0, rtol=1e-5, atol=1e-5)

  # Direct ufunc + direct Python eager call
  np.testing.assert_array_equal(dsl.imap(dsl.add, (3, 4)), np.arange(3)[:, None] + np.arange(4)[None, :])
  np.testing.assert_array_equal(
    dsl.imap(lambda i, j: i * 10 + j, (3, 4)), np.arange(3)[:, None] * 10 + np.arange(4)[None, :]
  )


def test_polymorphic_dsl_min_max_and_pure_dsl_attention():
  @dsl.kernel
  def causal_attention_dsl(q, k, v):
    causal = dsl.imap(lambda i, j: i >= j, (q.shape[0], k.shape[0]))
    s = dsl.where(causal, (q @ k.T) * dsl.f32(0.125), dsl.f32(-1e9))
    e = dsl.exp(s - dsl.max(s, axis=-1, keepdims=True))
    return (e / dsl.sum(e, axis=-1, keepdims=True)) @ v

  @dsl.kernel
  def minmax_ops(x, y):
    return dsl.min(x, y), dsl.max(x, y), dsl.min(x, axis=1), dsl.reduce(dsl.max, y, axis=0)

  rng = np.random.default_rng(13)
  q = rng.normal(size=(32, 16)).astype(np.float32) * 0.2
  k = rng.normal(size=(32, 16)).astype(np.float32) * 0.2
  v = rng.normal(size=(32, 16)).astype(np.float32) * 0.2
  rep = dsl.show_rewrite(causal_attention_dsl, q, k, v, backend="multicore")
  assert "flash_monoid" in rep.monoidal_plans
  assert any("Law 5" in law for law in rep.laws_triggered)

  mask = np.tril(np.ones((32, 32), dtype=bool))
  s_ref = np.where(mask, (q @ k.T) * np.float32(0.125), np.float32(-1e9))
  e_ref = np.exp(s_ref - np.max(s_ref, axis=-1, keepdims=True))
  ref = (e_ref / np.sum(e_ref, axis=-1, keepdims=True)) @ v
  np.testing.assert_allclose(causal_attention_dsl(q, k, v, _backend="multicore"), ref, rtol=1e-5, atol=1e-5)

  x = rng.normal(size=(5, 7)).astype(np.float32)
  y = rng.normal(size=(5, 7)).astype(np.float32)
  mn_ew, mx_ew, mn_ax, mx_red = minmax_ops(x, y, _backend="multicore")
  np.testing.assert_allclose(mn_ew, np.minimum(x, y))
  np.testing.assert_allclose(mx_ew, np.maximum(x, y))
  np.testing.assert_allclose(mn_ax, np.min(x, axis=1))
  np.testing.assert_allclose(mx_red, np.max(y, axis=0))
  # Plain Python calls
  np.testing.assert_allclose(dsl.min(x, y), np.minimum(x, y))
  np.testing.assert_allclose(dsl.max(x, axis=-1, keepdims=True), np.max(x, axis=-1, keepdims=True))


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_gallery_dsl_kernels_and_nanogpt_smoke(backend):
  from examples.algebraic_gallery import (
    exact_1nn,
    gbdt_best_split,
    nearest_fold,
    nearest_numpy,
    smooth_fold,
    smooth_numpy,
    tropical_min_plus,
  )

  rng = np.random.default_rng(21)
  a = rng.normal(size=(8, 6)).astype(np.float32)
  b = rng.normal(size=(6, 9)).astype(np.float32)
  np.testing.assert_allclose(
    tropical_min_plus(a, b, _backend=backend),
    np.min(a[:, :, None] + b[None, :, :], axis=1),
    rtol=1e-5,
    atol=1e-5,
  )

  q = rng.normal(size=(12, 4)).astype(np.float32)
  r = rng.normal(size=(35, 4)).astype(np.float32)
  r[2] = r[19] = q[0]
  obs = rng.normal(size=(35,)).astype(np.float32)
  exp_nn = nearest_numpy(q, r, _backend=backend)
  np.testing.assert_array_equal(exact_1nn(q, r, _backend=backend), exp_nn)
  np.testing.assert_array_equal(nearest_fold(q, r, _backend=backend), exp_nn)
  np.testing.assert_allclose(
    smooth_fold(q, r, obs, _backend=backend), smooth_numpy(q, r, obs, _backend=backend), rtol=1e-5, atol=1e-5
  )

  bins = rng.integers(0, 16, size=(200,), dtype=np.int64)
  g = rng.normal(size=(200,)).astype(np.float32)
  h = rng.uniform(0.1, 1.0, size=(200,)).astype(np.float32)
  b_bin, b_gain = gbdt_best_split(bins, g, h, 16, np.float32(1.0), _backend=backend)
  assert 0 <= int(b_bin) < 16 and np.isfinite(float(b_gain))
