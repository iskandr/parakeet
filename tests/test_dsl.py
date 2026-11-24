"""Tests for parakeet.dsl explicit data-parallel DSL & algebraic rewrite inspector."""

from __future__ import annotations

import numpy as np

import parakeet
from parakeet import dsl


@dsl.kernel
def _dsl_flash(q, k, v):
  causal = dsl.imap(lambda ij: ij[0] >= ij[1], (q.shape[0], k.shape[0]))
  s = np.where(causal, (q @ k.T) * np.float32(0.125), np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


@dsl.kernel
def _dsl_mla(q, c_kv, w_uk, w_uv):
  k = c_kv @ w_uk
  v = c_kv @ w_uv
  s = (q @ k.T) * np.float32(0.125)
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


@dsl.kernel
def _dsl_ssd(q, k, v, a):
  T = q.shape[0]
  out = np.empty((T, v.shape[1]), dtype=np.float32)
  S = np.zeros((k.shape[1], v.shape[1]), dtype=np.float32)
  for t in range(T):
    S = a[t] * S + k[t][:, None] * v[t][None, :]
    out[t] = q[t] @ S
  return out


@dsl.kernel
def _dsl_gbdt_hist(bins, grads, nbins):
  return dsl.iscatter(nbins, bins, grads)


def test_dsl_show_rewrite_reports_all_algebraic_laws():
  rng = np.random.default_rng(0)
  q = rng.standard_normal((32, 32)).astype(np.float32)
  k = rng.standard_normal((32, 32)).astype(np.float32)
  v = rng.standard_normal((32, 32)).astype(np.float32)

  r_flash = dsl.show_rewrite(_dsl_flash, q, k, v)
  assert "flash_monoid" in r_flash.monoidal_plans
  assert any("Law 1" in x for x in r_flash.laws_triggered)
  assert any("Law 5" in x for x in r_flash.laws_triggered)
  assert "Parakeet Algebraic Rewrite Report" in r_flash.summary()

  c_kv = rng.standard_normal((128, 16)).astype(np.float32)
  wuk = rng.standard_normal((16, 32)).astype(np.float32)
  wuv = rng.standard_normal((16, 32)).astype(np.float32)
  r_mla = dsl.show_rewrite(_dsl_mla, q[:2], c_kv, wuk, wuv)
  assert any("Law 3" in x for x in r_mla.laws_triggered)
  assert any("Law 1" in x for x in r_mla.laws_triggered)

  a = rng.uniform(0.8, 0.99, size=(128,)).astype(np.float32)
  r_ssd = dsl.show_rewrite(
    _dsl_ssd,
    rng.standard_normal((128, 16)).astype(np.float32),
    rng.standard_normal((128, 16)).astype(np.float32),
    rng.standard_normal((128, 16)).astype(np.float32),
    a,
  )
  assert any("Law 4" in x for x in r_ssd.laws_triggered)

  bins = rng.integers(0, 16, size=(1000,), dtype=np.int64)
  grads = rng.standard_normal((1000,)).astype(np.float32)
  r_hist = dsl.show_rewrite(_dsl_gbdt_hist, bins, grads, 16)
  assert r_hist.after_scatters >= 1
  assert any("IndexScatter" in x for x in r_hist.laws_triggered)
  np.testing.assert_allclose(_dsl_gbdt_hist(bins, grads, 16), np.bincount(bins, weights=grads, minlength=16), rtol=1e-5, atol=1e-5)

  assert parakeet.dsl is dsl


def test_dsl_show_rewrite_reliable_on_warm_cache_and_grad():
  """Calling show_rewrite repeatedly, after ordinary execution, and on grad()
  wrappers must always return non-empty before/after IR without executing."""
  def sq_sum(x):
    return np.sum(x * x)

  x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
  # Warm the specialization cache first via ordinary execution
  jitted = parakeet.jit(sq_sum)
  np.testing.assert_allclose(jitted(x), 30.0)

  r1 = dsl.show_rewrite(sq_sum, x, backend="c")
  r2 = dsl.show_rewrite(sq_sum, x, backend="c")
  assert r1.before_ir and r1.after_ir
  assert r2.before_ir and r2.after_ir
  assert r1.before_maps == r2.before_maps and r1.after_maps == r2.after_maps

  # Repeated inspection on grad(sq_sum) (compile-only, never empty on 2nd call)
  g = parakeet.grad(sq_sum)
  np.testing.assert_allclose(g(x), 2.0 * x)
  rg1 = dsl.show_rewrite(g, x, backend="c")
  rg2 = dsl.show_rewrite(g, x, backend="c")
  assert rg1.before_ir and rg1.after_ir
  assert rg2.before_ir and rg2.after_ir

  # Cost-guarded reassociation is explicitly marked as cost-guarded in IR
  rng = np.random.default_rng(1)
  q = rng.standard_normal((2, 32)).astype(np.float32)
  c_kv = rng.standard_normal((128, 16)).astype(np.float32)
  wuk = rng.standard_normal((16, 32)).astype(np.float32)
  wuv = rng.standard_normal((16, 32)).astype(np.float32)
  r_mla = dsl.show_rewrite(_dsl_mla, q, c_kv, wuk, wuv)
  assert any("cost-guarded" in law for law in r_mla.laws_triggered), r_mla.laws_triggered

