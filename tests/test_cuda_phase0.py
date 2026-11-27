"""Tests for Issue #104 Phase 0: headerless inline-PTX mma.sync TF32 Tensor Cores,
register-direct epilogues, and persistent CUBIN/PTX disk cache."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import parakeet as par
from parakeet.backends.cuda import _CudaBinaryCache


def test_cuda_binary_cache_roundtrip_and_isolation(tmp_path):
  c1 = _CudaBinaryCache("sm_89", "nvrtc_13_0", cache_root=tmp_path)
  k1 = c1.key_for('extern "C" __global__ void pk_kernel() {}')
  assert c1.get(k1) is None
  assert c1.misses == 1 and c1.hits == 0

  payload = b"\x7fELF_FAKE_CUBIN_PAYLOAD_1234"
  c1.put(k1, payload)
  assert c1.get(k1) == payload
  assert c1.hits == 1

  # Second instance (simulating a new process) hits the same on-disk CUBIN immediately
  c2 = _CudaBinaryCache("sm_89", "nvrtc_13_0", cache_root=tmp_path)
  assert c2.get(c2.key_for('extern "C" __global__ void pk_kernel() {}')) == payload
  assert c2.hits == 1 and c2.misses == 0

  # Different SM arch or NVRTC version produces a distinct cache key
  c_arch = _CudaBinaryCache("sm_80", "nvrtc_13_0", cache_root=tmp_path)
  assert c_arch.key_for('extern "C" __global__ void pk_kernel() {}') != k1
  assert c_arch.get(c_arch.key_for('extern "C" __global__ void pk_kernel() {}')) is None

  c_ver = _CudaBinaryCache("sm_89", "nvrtc_12_6", cache_root=tmp_path)
  assert c_ver.key_for('extern "C" __global__ void pk_kernel() {}') != k1


@pytest.mark.skipif(sys.platform == "win32", reason="CUDA simulator requires POSIX C++")
def test_inline_ptx_mma_all_orientations_and_register_epilogue(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.api import _backends
  from parakeet.backends import cuda

  cuda.CudaDevice._instance = None
  _backends.pop("cuda", None)

  @par.jit(backend="cuda")
  def gemm_nn(a, b):
    return a @ b

  @par.jit(backend="cuda")
  def gemm_nt(a, b):
    return a @ b.T

  @par.jit(backend="cuda")
  def gemm_tn(a, b):
    return a.T @ b

  @par.jit(backend="cuda")
  def gemm_tt(a, b):
    return a.T @ b.T

  @par.jit(backend="cuda")
  def gemm_bias_tanh(a, b, bias):
    return np.tanh(a @ b + bias)

  @par.jit(backend="cuda")
  def batched_gemm(a, b):
    return a @ b

  rng = np.random.default_rng(104)
  with par.options(tensor_cores=True, blas=False, device_threshold=1, device_fallback="error"):
    # All 4 NN/NT/TN/TT orientations on both aligned (64x64) and odd tail (53x41x37) shapes
    for m, k, n in ((64, 64, 64), (53, 41, 37)):
      a = rng.standard_normal((m, k)).astype(np.float32) * np.float32(0.25)
      b = rng.standard_normal((k, n)).astype(np.float32) * np.float32(0.25)
      bt = np.ascontiguousarray(b.T)
      at = np.ascontiguousarray(a.T)
      bias = rng.standard_normal((n,)).astype(np.float32) * np.float32(0.1)

      for fn, args, ref in (
        (gemm_nn, (a, b), a @ b),
        (gemm_nt, (a, bt), a @ bt.T),
        (gemm_tn, (at, b), at.T @ b),
        (gemm_tt, (at, bt), at.T @ bt.T),
        (gemm_bias_tanh, (a, b, bias), np.tanh(a @ b + bias)),
      ):
        got = fn(*args)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=2e-3)
        ksrc = "\n".join(fn.specialize(*args).kernels)
        assert "#include <mma.h>" not in ksrc
        assert "pk_mma_m16n8k8_tf32(" in ksrc
        assert "store_matrix_sync" not in ksrc

    # Batched 3D GEMM
    ba = rng.standard_normal((3, 35, 48)).astype(np.float32) * np.float32(0.25)
    bb = rng.standard_normal((3, 48, 29)).astype(np.float32) * np.float32(0.25)
    np.testing.assert_allclose(batched_gemm(ba, bb), ba @ bb, rtol=1e-3, atol=2e-3)
