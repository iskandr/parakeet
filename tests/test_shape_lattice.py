"""Tests for the finite shape-bucket & divisibility abstract domain,
demand-driven specialization guards, and Hielscher interior f^k vs straggler f
GPU tile specialization."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import parakeet as par
from parakeet.shape_lattice import TOP, DimInfo, classify_dim, describe_dim, pow2_align


def test_dim_info_buckets_and_soundness():
  # Exact small constants and powers of 2 up to 64
  for n in (0, 1, 2, 3, 4, 8, 16, 32, 64):
    d = classify_dim(n)
    assert d.exact == n
    assert d.matches(n)
    assert d.align == pow2_align(n)

  # Intermediate power-of-2 buckets preserve exact interval + divisibility
  assert classify_dim(6) == DimInfo(5, 7, 2)
  assert classify_dim(12) == DimInfo(9, 15, 4)
  assert classify_dim(24) == DimInfo(17, 31, 8)
  assert classify_dim(48) == DimInfo(33, 63, 16)
  assert classify_dim(65) == DimInfo(65, 127, 1)
  assert classify_dim(96) == DimInfo(65, 127, 32)

  # All large multiples of 128 collapse into ONE bucket (zero recompilation churn)
  big = classify_dim(128)
  assert big == DimInfo(128, None, 128)
  for n in (128, 256, 384, 512, 1024, 2048, 4096, 8192):
    assert classify_dim(n) == big
    assert big.matches(n)
  assert not big.matches(64)
  assert not big.matches(129)
  assert not big.matches(192)  # only divisible by 64, not 128

  # Monotonicity: a stronger alignment still matches a weaker guard
  assert DimInfo(65, 127, 4).matches(100)
  assert DimInfo(65, 127, 4).matches(104)
  assert not DimInfo(65, 127, 4).matches(102)

  # Human-readable formatting
  assert describe_dim(classify_dim(64)) == "64"
  assert describe_dim(classify_dim(128)) == ">=128%128==0"
  assert describe_dim(classify_dim(65)) == "65..127"


def test_dim_info_abstract_arithmetic_soundness():
  samples = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 65, 96, 128, 192, 256, 512]
  for a in samples:
    da = classify_dim(a)
    for b in samples:
      db = classify_dim(b)
      assert da.add(db).matches(a + b)
      assert da.mul(db).matches(a * b)
      assert da.imin(db).matches(min(a, b))
      assert da.imax(db).matches(max(a, b))
      if a >= b:
        assert da.sub(db).matches(a - b)
      if b > 0:
        assert da.floordiv_const(b).matches(a // b)

  # Flattening (B=32, T=128) -> (>=4096, div128) and head splitting C//4
  bt = classify_dim(32).mul(classify_dim(128))
  assert bt == DimInfo(4096, None, 128)
  head_d = classify_dim(256).floordiv_const(4)
  assert head_d == DimInfo(32, None, 32)
  assert head_d.divisible_by(16) and head_d.divisible_by(32)

  # Meet sharpens bounds and alignment when two dims are proven equal
  m = DimInfo(65, 127, 1).meet(DimInfo(64, None, 32))
  assert m == DimInfo(65, 127, 32)
  assert m.matches(96)


def test_demand_driven_guards_do_not_recompile_unrelated_shapes():
  @par.jit(backend="c")
  def scale_add(x):
    return x * 2.0 + 1.0

  for n in (3, 7, 15, 64, 129):
    x = np.arange(n, dtype=np.float32)
    np.testing.assert_allclose(scale_add(x), x * 2.0 + 1.0)
  assert len(scale_add.specializations) == 1
  assert scale_add.specializations[0].dim_guards == ()


@pytest.mark.skipif(sys.platform == "win32", reason="CUDA simulator requires POSIX C++")
def test_cuda_sim_shape_specialized_wmma_and_cache_reuse(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.api import _backends
  from parakeet.backends import cuda

  cuda.CudaDevice._instance = None
  _backends.pop("cuda", None)

  @par.jit(backend="cuda")
  def mm(a, b):
    return a @ b

  with par.options(tensor_cores=True, shape_specialization=True):
    # 1. N=64 -> shape-adapted 64x64x32 WMMA with branchless interior cp.async + epilogue
    rng = np.random.default_rng(0)
    a1 = rng.standard_normal((128, 128)).astype(np.float32)
    b1 = rng.standard_normal((128, 64)).astype(np.float32)
    out1 = mm(a1, b1)
    np.testing.assert_allclose(out1, a1 @ b1, rtol=1e-2, atol=2e-2)
    spec1 = mm.specialize(a1, b1)
    assert spec1.dim_guards  # recorded on-demand guards for M, K, N
    ksrc1 = "\n".join(spec1.kernels)
    assert "pk_mma_m16n8k8_tf32(" in ksrc1
    assert "#include <mma.h>" not in ksrc1
    assert "store_matrix_sync" not in ksrc1
    assert "pk_cp_async_16(pk_sp, pk_gp);" in ksrc1
    # Interior f^k fast path elides slow/fast bounds checks and 15ULL alignment check
    assert "15ULL" not in ksrc1
    assert "< pk_M &&" not in ksrc1
    # Confirm 64x64x32 tile chosen (128 threads, BK=32)
    tiled_meta = next(k.device.extra["tiled"] for k in spec1.module.kernels if k.device and "tiled" in k.device.extra)
    assert (tiled_meta["bm"], tiled_meta["bn"], tiled_meta["bk"]) == (64, 64, 32)
    assert len(mm.specializations) == 1

    # 2. Larger aligned (256, 256) @ (256, 64) REUSES the exact same specialization!
    a2 = rng.standard_normal((256, 256)).astype(np.float32)
    b2 = rng.standard_normal((256, 64)).astype(np.float32)
    out2 = mm(a2, b2)
    np.testing.assert_allclose(out2, a2 @ b2, rtol=1e-2, atol=3e-2)
    assert len(mm.specializations) == 1

    # 3. N=128 -> misses N=64 guard, compiles 128x128x16 interior f^k specialization
    a3 = rng.standard_normal((128, 128)).astype(np.float32)
    b3 = rng.standard_normal((128, 128)).astype(np.float32)
    out3 = mm(a3, b3)
    np.testing.assert_allclose(out3, a3 @ b3, rtol=1e-2, atol=2e-2)
    assert len(mm.specializations) == 2
    spec3 = mm.specialize(a3, b3)
    tiled_meta3 = next(k.device.extra["tiled"] for k in spec3.module.kernels if k.device and "tiled" in k.device.extra)
    assert (tiled_meta3["bm"], tiled_meta3["bn"], tiled_meta3["bk"]) == (128, 128, 16)
    assert "15ULL" not in "\n".join(spec3.kernels)

    # 4. Odd straggler shape (37, 29) @ (29, 19) -> misses alignment guards, runs guarded f path
    a4 = rng.standard_normal((37, 29)).astype(np.float32)
    b4 = rng.standard_normal((29, 19)).astype(np.float32)
    out4 = mm(a4, b4)
    np.testing.assert_allclose(out4, a4 @ b4, rtol=1e-2, atol=2e-2)
    assert len(mm.specializations) == 3
    spec4 = mm.specialize(a4, b4)
    ksrc4 = "\n".join(spec4.kernels)
    assert "15ULL" in ksrc4
    assert "< pk_M &&" in ksrc4

    # 5. Sliced (misaligned offset) input: layout="S" -> distinct specialization, stays sound
    a5 = rng.standard_normal((130, 130)).astype(np.float32)[1:129, 1:129]
    b5 = rng.standard_normal((130, 66)).astype(np.float32)[1:129, 1:65]
    out5 = mm(a5, b5)
    np.testing.assert_allclose(out5, a5 @ b5, rtol=1e-2, atol=2e-2)


@pytest.mark.skipif(sys.platform == "win32", reason="CUDA simulator requires POSIX C++")
def test_cuda_sim_simt_interior_and_warp_row_unroll(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.api import _backends
  from parakeet.backends import cuda

  cuda.CudaDevice._instance = None
  _backends.pop("cuda", None)

  @par.jit(backend="cuda")
  def mm_simt(a, b):
    return a @ b

  @par.jit(backend="cuda")
  def ln_row(x, g, b):
    mu = np.mean(x, axis=-1, keepdims=True)
    d = x - mu
    inv = np.float32(1.0) / np.sqrt(np.mean(d * d, axis=-1, keepdims=True) + np.float32(1e-5))
    return d * inv * g + b

  @par.jit(backend="cuda")
  def col_sum(x):
    return np.sum(x, axis=0)

  rng = np.random.default_rng(1)
  with par.options(tensor_cores=False, shape_specialization=True):
    # SIMT tiled GEMM on divisible shape -> compile-time constant BK loop, no pk_klim
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = rng.standard_normal((64, 64)).astype(np.float32)
    np.testing.assert_allclose(mm_simt(a, b), a @ b, rtol=1e-4, atol=1e-4)
    spec_simt = mm_simt.specialize(a, b)
    k_simt = "\n".join(spec_simt.kernels)
    simt_meta = next(
      k.device.extra["tiled"] for k in spec_simt.module.kernels if k.device and "tiled" in k.device.extra
    )
    assert "pk_klim" not in k_simt
    assert f"pk_tk < {simt_meta['bk']}; pk_tk++" in k_simt

    # Straggler SIMT shape -> keeps pk_klim and stays exact
    ao = rng.standard_normal((37, 29)).astype(np.float32)
    bo = rng.standard_normal((29, 19)).astype(np.float32)
    np.testing.assert_allclose(mm_simt(ao, bo), ao @ bo, rtol=1e-4, atol=1e-4)
    assert "pk_klim" in "\n".join(mm_simt.specialize(ao, bo).kernels)

    # Warp row pipeline on (32, 64) -> N%8==0 elides row guard, K==64 unrolls 2 iters
    x = rng.standard_normal((32, 64)).astype(np.float32)
    g = rng.standard_normal(64).astype(np.float32)
    bv = rng.standard_normal(64).astype(np.float32)
    got = ln_row(x, g, bv)
    ref_mu = x.mean(axis=-1, keepdims=True)
    ref = (x - ref_mu) / np.sqrt(((x - ref_mu) ** 2).mean(axis=-1, keepdims=True) + 1e-5) * g + bv
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    k_ln = "\n".join(ln_row.specialize(x, g, bv).kernels)
    assert "for (int pk_ki = 0; pk_ki < 2; ++pk_ki)" in k_ln
    assert ">= pk_N) return;" not in k_ln

    # Non-aligned warp row pipeline -> keeps guards and matches
    xo = rng.standard_normal((13, 37)).astype(np.float32)
    go = rng.standard_normal(37).astype(np.float32)
    bo2 = rng.standard_normal(37).astype(np.float32)
    got_o = ln_row(xo, go, bo2)
    ref_mu_o = xo.mean(axis=-1, keepdims=True)
    ref_o = (xo - ref_mu_o) / np.sqrt(((xo - ref_mu_o) ** 2).mean(axis=-1, keepdims=True) + 1e-5) * go + bo2
    np.testing.assert_allclose(got_o, ref_o, rtol=1e-4, atol=1e-4)
    assert ">= pk_N) return;" in "\n".join(ln_row.specialize(xo, go, bo2).kernels)

    # split_gpu_reduce dead-branch elimination: (4096, 64) -> always_split=True -> single atomic kernel!
    xs = rng.standard_normal((4096, 64)).astype(np.float32)
    np.testing.assert_allclose(col_sum(xs), xs.sum(axis=0), rtol=1e-3, atol=1e-3)
    spec_s = col_sum.specialize(xs)
    assert len(spec_s.kernels) == 1
    assert "atomicAdd" in spec_s.kernels[0]

  # Disabling shape_specialization yields zero dim_guards
  with par.options(shape_specialization=False):
    assert mm_simt.specialize(a, b).dim_guards == ()


def test_dim_info_top_sanity():
  assert TOP.is_top
  assert TOP.matches(0) and TOP.matches(1) and TOP.matches(999999)
