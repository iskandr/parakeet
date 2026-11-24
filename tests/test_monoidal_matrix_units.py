"""Tests for 64x64x16 hardware matrix-unit monoidal GPU kernels
(CUDA ``nvcuda::wmma`` TF32 Tensor Cores and Apple Metal ``simdgroup_matrix``)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import parakeet as pk
from examples.algebraic_gallery import (
  causal_flash_attention,
  cut_cross_entropy,
  gqa_attention,
  sliding_window_attention,
)


@pk.jit
def _bool_masked_attention(q, k, v, mask):
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = np.where(mask, (q @ k.T) * scale, np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


@pk.jit
def _plain_attention(q, k, v):
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = (q @ k.T) * scale
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


def _cuda_available() -> bool:
  return os.environ.get("PARAKEET_CUDA_SIMULATE") == "1" or pk.available_backends().get("cuda") == "ok"


def _metal_available() -> bool:
  return sys.platform == "darwin" and pk.available_backends().get("metal") == "ok"


def _ref_attn(q, k, v, mask=None, fill=-1e9):
  s = (q.astype("d") @ k.astype("d").swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
  if mask is not None:
    s = np.where(mask, s, fill)
  e = np.exp(s - s.max(-1, keepdims=True))
  return ((e / e.sum(-1, keepdims=True)) @ v.astype("d")).astype(np.float32)


def _run_matrix_unit_checks(backend: str, *, tensor_cores: bool, marker: str) -> None:
  rng = np.random.default_rng(19)
  opts = {"backend": backend, "blas": False, "device_fallback": "error", "device_threshold": 1}
  if backend == "cuda":
    opts["tensor_cores"] = tensor_cores

  with pk.options(**opts):
    # 1. Causal attention: non-multiple of 64, K!=D, and D > MAX_REG_D (140 > 128)
    for shape_q, shape_kv in [
      ((1, 2, 73, 64), (1, 2, 73, 64)),
      ((1, 1, 130, 37), (1, 1, 130, 37)),
      ((1, 1, 65, 140), (1, 1, 65, 140)),
    ]:
      q = rng.standard_normal(shape_q, dtype=np.float32) * np.float32(0.2)
      k = rng.standard_normal(shape_kv, dtype=np.float32) * np.float32(0.2)
      v = rng.standard_normal(shape_kv, dtype=np.float32) * np.float32(0.2)
      spec = causal_flash_attention.specialize(q, k, v)
      assert any(marker in src for src in spec.kernels), f"expected {marker} in generated kernel"
      out = pk.to_host(causal_flash_attention(*pk.to_device((q, k, v), backend=backend)))
      T = shape_q[-2]
      ref = _ref_attn(q, k, v, mask=np.tril(np.ones((T, T), dtype=bool)))
      np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    # 2. Sliding-window attention (Law 5 64x64 tile skipping)
    q = rng.standard_normal((1, 2, 135, 32), dtype=np.float32) * np.float32(0.2)
    k = rng.standard_normal((1, 2, 135, 32), dtype=np.float32) * np.float32(0.2)
    v = rng.standard_normal((1, 2, 135, 32), dtype=np.float32) * np.float32(0.2)
    out = pk.to_host(sliding_window_attention(*pk.to_device((q, k, v), backend=backend), 20))
    idx = np.arange(135)
    sw_mask = (idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) <= 20)
    np.testing.assert_allclose(out, _ref_attn(q, k, v, mask=sw_mask), rtol=1e-3, atol=1e-3)

    # 3. Runtime boolean-mask attention (bit-packed 64x64 mask with all-false tiles!)
    M, N, K, D = 70, 140, 32, 48
    q2 = rng.standard_normal((M, K), dtype=np.float32) * np.float32(0.2)
    k2 = rng.standard_normal((N, K), dtype=np.float32) * np.float32(0.2)
    v2 = rng.standard_normal((N, D), dtype=np.float32) * np.float32(0.2)
    bmask = np.zeros((M, N), dtype=bool)
    bmask[:, :64] = rng.random((M, 64)) > 0.3
    bmask[:, 0] = True  # ensure every row has at least one valid entry
    # Leave cols 64..127 completely False to exercise all-false tile skipping
    bmask[:, 128:] = rng.random((M, N - 128)) > 0.5
    spec_bm = _bool_masked_attention.specialize(q2, k2, v2, bmask)
    assert any(marker in src for src in spec_bm.kernels)
    out_bm = pk.to_host(_bool_masked_attention(*pk.to_device((q2, k2, v2, bmask), backend=backend)))
    np.testing.assert_allclose(out_bm, _ref_attn(q2, k2, v2, mask=bmask), rtol=1e-3, atol=1e-3)

    # 4. Split-K FlashDecoding (small T_q=4, long T_kv=260)
    qd = rng.standard_normal((4, 64), dtype=np.float32) * np.float32(0.2)
    kd = rng.standard_normal((260, 64), dtype=np.float32) * np.float32(0.2)
    vd = rng.standard_normal((260, 48), dtype=np.float32) * np.float32(0.2)
    out_dec = pk.to_host(_plain_attention(*pk.to_device((qd, kd, vd), backend=backend)))
    np.testing.assert_allclose(out_dec, _ref_attn(qd, kd, vd), rtol=1e-3, atol=1e-3)

    # 5. GQA shared-KV head-group tiling
    qg = rng.standard_normal((1, 2, 4, 35, 32), dtype=np.float32) * np.float32(0.2)
    kg = rng.standard_normal((1, 2, 1, 70, 32), dtype=np.float32) * np.float32(0.2)
    vg = rng.standard_normal((1, 2, 1, 70, 32), dtype=np.float32) * np.float32(0.2)
    out_gqa = pk.to_host(gqa_attention(*pk.to_device((qg, kg, vg), backend=backend)))
    np.testing.assert_allclose(out_gqa, _ref_attn(qg, kg, vg), rtol=1e-3, atol=1e-3)

    # 6. Cut Cross-Entropy (monoid_lse with non-multiple of 64 M, K, V and negative targets)
    h = rng.standard_normal((71, 35), dtype=np.float32) * np.float32(0.2)
    w = rng.standard_normal((35, 197), dtype=np.float32) * np.float32(0.2)
    tgt = rng.integers(197, size=71, dtype=np.int64)
    tgt[0] = -1  # negative target index wrap check
    cce = pk.jit(cut_cross_entropy)
    spec_cce = cce.specialize(h, w, tgt)
    assert any(marker in src for src in spec_cce.kernels)
    loss = cce(*pk.to_device((h, w, tgt), backend=backend))
    logits = h.astype("d") @ w.astype("d")
    z = logits - logits.max(1, keepdims=True)
    ref_loss = np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(71), tgt])
    np.testing.assert_allclose(float(loss), float(ref_loss), rtol=1e-3, atol=1e-3)


def test_monoidal_wmma_cuda():
  if not _cuda_available():
    pytest.skip("CUDA backend/simulator unavailable")
  _run_matrix_unit_checks("cuda", tensor_cores=True, marker="wmma")

  # Confirm tensor_cores=False keeps exact FP32 SIMT path (no wmma)
  q = np.zeros((64, 64), dtype=np.float32)
  with pk.options(backend="cuda", blas=False, device_fallback="error", device_threshold=1, tensor_cores=False):
    spec = _plain_attention.specialize(q, q, q)
    assert not any("wmma" in src for src in spec.kernels)


def test_monoidal_simdgroup_metal():
  if not _metal_available():
    pytest.skip("Metal backend unavailable")
  _run_matrix_unit_checks("metal", tensor_cores=False, marker="simdgroup_float8x8")
