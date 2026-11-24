"""Tests for Algebraic Law 3 (Bilinear Contraction Reassociation, MLA Weight
Absorption, GQA Head-Group Tiling) and Law 4 (Chunkwise State-Space Duality &
Parallel Affine Prefix Scan / TileScan).
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import parakeet
from parakeet import value_and_grad
from parakeet.syntax import ir
from parakeet.transforms import pipeline


def _inspect_ir_at(fn, args, stage: str, probe, **opts):
  """Run ``probe(ir_fn)`` immediately after ``stage`` (before later passes mutate it)."""
  result: dict[str, object] = {}
  orig_run = pipeline.run

  def hooked(ir_fn, options, trace=None):
    def tr(name, f):
      if name == stage:
        result["val"] = probe(f)
        result["dump"] = ir.fmt_function(f)
      if trace is not None:
        trace(name, f)

    orig_run(ir_fn, options, trace=tr)

  pipeline.run = hooked
  try:
    with parakeet.options(**opts):
      fn.specialize(*args)
  finally:
    pipeline.run = orig_run
  return result["val"], result["dump"]


def _count_monoid_plans(ir_fn: ir.Function, kind: str) -> int:
  n = 0
  for s in ir.walk_stmts(ir_fn.body):
    if isinstance(s, ir.IndexTile) and s.monoidal_tile is not None and s.monoidal_tile.kind == kind:
      n += 1
  return n


# ---------------------------------------------------------------------------
# Law 3: General Bilinear Contraction Reassociation (LoRA / MLP / Chains)
# ---------------------------------------------------------------------------


@parakeet.jit
def _lora_chain(x, a, b):
  return (x @ a) @ b


@parakeet.jit
def _right_chain(x, a, b):
  return x @ (a @ b)


@parakeet.jit
def _lora_loss(x, a, b):
  return np.sum(((x @ a) @ b) ** np.float32(2.0))


def test_law3_bilinear_reassociation_lora_and_ad():
  rng = np.random.default_rng(7)
  # Case A: large batch M=128, small N=8 -> X @ (A @ B) is strictly cheaper than (X @ A) @ B
  x_big = rng.standard_normal((128, 32)).astype(np.float32)
  a_big = rng.standard_normal((32, 32)).astype(np.float32)
  b_big = rng.standard_normal((32, 8)).astype(np.float32)

  # Case B: small batch M=4, low rank r=8 -> (X @ A) @ B is strictly cheaper than X @ (A @ B)
  x_sm = rng.standard_normal((4, 64)).astype(np.float32)
  a_sm = rng.standard_normal((64, 8)).astype(np.float32)
  b_sm = rng.standard_normal((8, 64)).astype(np.float32)

  has_reassoc_if, dump = _inspect_ir_at(
    _lora_chain,
    (x_big, a_big, b_big),
    "contraction reassociation",
    lambda fn: any(
      isinstance(s, ir.If) and any(isinstance(t, ir.Alloc) and t.var.name.startswith("reassoc_m") for t in s.then)
      for s in ir.walk_stmts(fn.body)
    ),
    blas=False,
    verify=True,
  )
  assert has_reassoc_if, f"Expected Law 3 contraction reassociation If:\n{dump}"

  with parakeet.options(blas=False, verify=True):
    np.testing.assert_allclose(_lora_chain(x_big, a_big, b_big), (x_big @ a_big) @ b_big, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(_lora_chain(x_sm, a_sm, b_sm), (x_sm @ a_sm) @ b_sm, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(_right_chain(x_sm, a_sm, b_sm), x_sm @ (a_sm @ b_sm), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(_right_chain(x_big, a_big, b_big), x_big @ (a_big @ b_big), rtol=1e-4, atol=1e-4)

    # Reverse-mode AD still matches finite differences / exact analytical gradients
    val, (dx, da, db) = value_and_grad(_lora_loss, argnum=(0, 1, 2))(x_sm, a_sm, b_sm)
    y = (x_sm @ a_sm) @ b_sm
    dy = 2.0 * y
    np.testing.assert_allclose(val, np.sum(y**2), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dx, (dy @ b_sm.T) @ a_sm.T, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(da, x_sm.T @ (dy @ b_sm.T), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(db, (x_sm @ a_sm).T @ dy, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Law 3 + Law 1: DeepSeek MLA Weight Absorption + FlashMLA
# ---------------------------------------------------------------------------


@parakeet.jit
def _mla_attn(q, c_kv, w_uk, w_uv):
  k = c_kv @ w_uk
  v = c_kv @ w_uv
  s = (q @ k.T) * np.float32(0.125)
  m = np.max(s, axis=-1, keepdims=True)
  e = np.exp(s - m)
  p = e / np.sum(e, axis=-1, keepdims=True)
  return p @ v


def _mla_ref(q, c_kv, w_uk, w_uv):
  k = c_kv @ w_uk
  v = c_kv @ w_uv
  s = (q @ k.T) * np.float32(0.125)
  m = np.max(s, axis=-1, keepdims=True)
  e = np.exp(s - m)
  p = e / np.sum(e, axis=-1, keepdims=True)
  return p @ v


def test_law3_mla_weight_absorption_composes_with_flash_monoid():
  rng = np.random.default_rng(11)
  # Decode shape: T_q = 2, T_kv = 128, d_c = 16, d_h = 48 -> absorbed branch is ~20x fewer FLOPs!
  q_dec = rng.standard_normal((2, 48)).astype(np.float32)
  c_dec = rng.standard_normal((128, 16)).astype(np.float32)
  wuk_dec = rng.standard_normal((16, 48)).astype(np.float32)
  wuv_dec = rng.standard_normal((16, 48)).astype(np.float32)

  # Prefill shape: T_q = 64, T_kv = 64, d_c = 48, d_h = 16 -> decompressed branch is cheaper!
  q_pre = rng.standard_normal((64, 16)).astype(np.float32)
  c_pre = rng.standard_normal((64, 48)).astype(np.float32)
  wuk_pre = rng.standard_normal((48, 16)).astype(np.float32)
  wuv_pre = rng.standard_normal((48, 16)).astype(np.float32)

  n_flash, dump = _inspect_ir_at(
    _mla_attn,
    (q_dec, c_dec, wuk_dec, wuv_dec),
    "monoidal tiling",
    lambda fn: _count_monoid_plans(fn, "flash_monoid"),
    blas=False,
    verify=True,
  )
  # BOTH branches of the MLA cost-guard If are fused into flash_monoid by Law 1!
  assert n_flash >= 2, f"Expected Law 3 MLA absorption + Law 1 FlashAttention fusion in both branches:\n{dump}"

  with parakeet.options(blas=False, verify=True):
    np.testing.assert_allclose(
      _mla_attn(q_dec, c_dec, wuk_dec, wuv_dec), _mla_ref(q_dec, c_dec, wuk_dec, wuv_dec), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
      _mla_attn(q_pre, c_pre, wuk_pre, wuv_pre), _mla_ref(q_pre, c_pre, wuk_pre, wuv_pre), rtol=1e-4, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Law 3: GQA Head-Group Tiling
# ---------------------------------------------------------------------------


@parakeet.jit
def _gqa_attn(q, k, v):
  # q: (B, H_kv, G, T_q, D), k, v: (B, H_kv, 1, T_kv, D)
  s = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(0.1767767)
  m = np.max(s, axis=-1, keepdims=True)
  e = np.exp(s - m)
  p = e / np.sum(e, axis=-1, keepdims=True)
  return np.matmul(p, v)


def test_law3_gqa_head_group_tiling():
  rng = np.random.default_rng(19)
  q = rng.standard_normal((2, 2, 4, 3, 32)).astype(np.float32)
  k = rng.standard_normal((2, 2, 1, 96, 32)).astype(np.float32)
  v = rng.standard_normal((2, 2, 1, 96, 32)).astype(np.float32)

  batch_ranks, dump = _inspect_ir_at(
    _gqa_attn,
    (q, k, v),
    "monoidal tiling",
    lambda fn: [
      len(s.monoidal_tile.batch_shape)
      for s in ir.walk_stmts(fn.body)
      if isinstance(s, ir.IndexTile) and s.monoidal_tile is not None and s.monoidal_tile.kind == "flash_monoid"
    ],
    blas=False,
    verify=True,
  )
  assert batch_ranks, f"Expected flash_monoid for GQA:\n{dump}"
  # Group axis G was folded into m_dim -> batch_shape has rank 2 (B, H_kv), not 3 (B, H_kv, G)!
  assert batch_ranks[0] == 2, f"Expected GQA group axis folded, got batch_ranks={batch_ranks}"

  s = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(0.1767767)
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  ref = np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)
  with parakeet.options(blas=False, verify=True):
    np.testing.assert_allclose(_gqa_attn(q, k, v), ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Law 4: Chunkwise Parallel Affine Prefix Scan (IIR / EMA / Kalman / GBDT)
# ---------------------------------------------------------------------------


@parakeet.jit
def _affine_scan_1d(a, b, s0):
  T = a.shape[0]
  out = np.empty_like(b)
  s = s0
  for t in range(T):
    s = a[t] * s + b[t]
    out[t] = s
  return out, s


@parakeet.jit
def _affine_scan_exclusive_1d(a, b, s0):
  T = a.shape[0]
  out = np.empty_like(b)
  s = s0
  for t in range(T):
    out[t] = s
    s = a[t] * s + b[t]
  return out, s


@parakeet.jit
def _affine_scan_batched(a, b):
  B, T = a.shape
  out = np.empty_like(b)
  for c in range(B):
    s = np.float32(0.0)
    for t in range(T):
      s = a[c, t] * s + b[c, t]
      out[c, t] = s
  return out


def _affine_ref_1d(a, b, s0):
  out = np.empty_like(b)
  s = np.float32(s0)
  for t in range(len(a)):
    s = np.float32(a[t] * s + b[t])
    out[t] = s
  return out, s


def test_law4_affine_tile_scan_1d_and_batched():
  rng = np.random.default_rng(23)
  # Non-multiple of AFFINE_CHUNK=64 (T=213) first so IR inspection compiles fresh!
  for T in (213, 64, 17):
    a = rng.uniform(0.85, 0.99, size=(T,)).astype(np.float32)
    b = rng.standard_normal((T,)).astype(np.float32)
    s0 = np.float32(1.25)
    if T == 213:
      has_chunks, dump = _inspect_ir_at(
        _affine_scan_1d,
        (a, b, s0),
        "chunkwise scan",
        lambda fn: any(isinstance(s, ir.Alloc) and s.var.name.startswith("ts_chA") for s in ir.walk_stmts(fn.body)),
        blas=False,
        verify=True,
      )
      assert has_chunks, f"Expected Law 4 TileScan chunk buffers:\n{dump}"
    with parakeet.options(blas=False, verify=True):
      out, fin = _affine_scan_1d(a, b, s0)
    ref_out, ref_fin = _affine_ref_1d(a, b, s0)
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(fin, ref_fin, rtol=1e-4, atol=1e-4)

  # Exclusive affine scan (stores state BEFORE update) across a partial final chunk (T=129)
  a_ex = np.ones(129, dtype=np.float32)
  b_ex = np.ones(129, dtype=np.float32)
  has_ex_chunks, ex_dump = _inspect_ir_at(
    _affine_scan_exclusive_1d,
    (a_ex, b_ex, np.float32(0.0)),
    "chunkwise scan",
    lambda fn: any(isinstance(s, ir.Alloc) and s.var.name.startswith("ts_chA") for s in ir.walk_stmts(fn.body)),
    blas=False,
    verify=True,
  )
  assert has_ex_chunks, f"Expected TileScan for exclusive scan:\n{ex_dump}"
  with parakeet.options(backend="c", blas=False, verify=True):
    ex_out, ex_fin = _affine_scan_exclusive_1d(a_ex, b_ex, np.float32(0.0))
  np.testing.assert_allclose(ex_out, np.arange(129, dtype=np.float32), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(ex_fin, np.float32(129.0), rtol=1e-5, atol=1e-5)
  # Also verify non-trivial (a, b) with non-zero s0 on partial chunk T=135
  a_ex2 = rng.uniform(0.8, 0.99, size=(135,)).astype(np.float32)
  b_ex2 = rng.standard_normal((135,)).astype(np.float32)
  s0_ex2 = np.float32(2.5)
  with parakeet.options(blas=False, verify=True):
    ex2_out, ex2_fin = _affine_scan_exclusive_1d(a_ex2, b_ex2, s0_ex2)
  ref_ex2 = np.empty_like(b_ex2)
  s_ref = np.float32(s0_ex2)
  for t in range(135):
    ref_ex2[t] = s_ref
    s_ref = np.float32(a_ex2[t] * s_ref + b_ex2[t])
  np.testing.assert_allclose(ex2_out, ref_ex2, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(ex2_fin, s_ref, rtol=1e-4, atol=1e-4)

  # Batched 2D scan (e.g. multi-channel IIR / GBDT histogram prefix sums)
  a2 = rng.uniform(0.9, 1.0, size=(6, 195)).astype(np.float32)
  b2 = rng.standard_normal((6, 195)).astype(np.float32)
  ref2 = np.stack([_affine_ref_1d(a2[i], b2[i], np.float32(0.0))[0] for i in range(6)])
  with parakeet.options(blas=False, verify=True):
    np.testing.assert_allclose(_affine_scan_batched(a2, b2), ref2, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Law 4: Chunkwise State-Space Duality (Mamba-2 SSD / Gated Linear Attention)
# ---------------------------------------------------------------------------


@parakeet.jit
def _mamba2_ssd(q, k, v, a):
  T = q.shape[0]
  D = k.shape[1]
  V = v.shape[1]
  out = np.empty((T, V), dtype=np.float32)
  S = np.zeros((D, V), dtype=np.float32)
  for t in range(T):
    S = a[t] * S + k[t][:, None] * v[t][None, :]
    out[t] = q[t] @ S
  return out, S


def _mamba2_ssd_ref(q, k, v, a):
  T, D = k.shape
  V = v.shape[1]
  out = np.empty((T, V), dtype=np.float32)
  S = np.zeros((D, V), dtype=np.float32)
  for t in range(T):
    S = a[t] * S + np.outer(k[t], v[t])
    out[t] = q[t] @ S
  return out, S


def test_law4_mamba2_ssd_chunkwise_duality():
  rng = np.random.default_rng(31)
  # Test T=150 first so IR inspection compiles fresh, including exact zeros & negatives in decay a[t]!
  for T in (150, 64, 20):
    q = rng.standard_normal((T, 16)).astype(np.float32) * 0.25
    k = rng.standard_normal((T, 16)).astype(np.float32) * 0.25
    v = rng.standard_normal((T, 24)).astype(np.float32) * 0.25
    a = rng.uniform(0.7, 0.99, size=(T,)).astype(np.float32)
    if T == 150:
      a[17] = 0.0  # hard reset (tests division-free decay product!)
      a[90] = -0.5  # negative decay (oscillatory SSM!)
      has_ssd, dump = _inspect_ir_at(
        _mamba2_ssd,
        (q, k, v, a),
        "chunkwise scan",
        lambda fn: any(isinstance(s, ir.Alloc) and s.var.name.startswith("ssd_Yi") for s in ir.walk_stmts(fn.body)),
        blas=False,
        verify=True,
      )
      assert has_ssd, f"Expected Law 4 Mamba-2 SSD chunkwise duality:\n{dump}"
    with parakeet.options(blas=False, verify=True):
      out, s_fin = _mamba2_ssd(q, k, v, a)
    ref_out, ref_s = _mamba2_ssd_ref(q, k, v, a)
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(s_fin, ref_s, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# GPU Execution: CUDA Simulator & Apple Metal
# ---------------------------------------------------------------------------


def _run_gpu_laws34(backend: str):
  rng = np.random.default_rng(42)
  q_m = rng.standard_normal((2, 32)).astype(np.float32)
  c_m = rng.standard_normal((128, 16)).astype(np.float32)
  wuk = rng.standard_normal((16, 32)).astype(np.float32)
  wuv = rng.standard_normal((16, 32)).astype(np.float32)

  qg = rng.standard_normal((1, 2, 4, 2, 24)).astype(np.float32)
  kg = rng.standard_normal((1, 2, 1, 96, 24)).astype(np.float32)
  vg = rng.standard_normal((1, 2, 1, 96, 24)).astype(np.float32)

  aa = rng.uniform(0.85, 0.99, size=(160,)).astype(np.float32)
  bb = rng.standard_normal((160,)).astype(np.float32)

  qs = rng.standard_normal((140, 16)).astype(np.float32) * 0.25
  ks = rng.standard_normal((140, 16)).astype(np.float32) * 0.25
  vs = rng.standard_normal((140, 16)).astype(np.float32) * 0.25
  as_ = rng.uniform(0.75, 0.98, size=(140,)).astype(np.float32)

  with parakeet.options(backend=backend, blas=False, device_threshold=1, device_fallback="error", verify=True):
    mla_out = _mla_attn(q_m, c_m, wuk, wuv)
    gqa_out = _gqa_attn(qg, kg, vg)
    aff_out, aff_fin = _affine_scan_1d(aa, bb, np.float32(0.5))
    ssd_out, ssd_s = _mamba2_ssd(qs, ks, vs, as_)

  np.testing.assert_allclose(mla_out, _mla_ref(q_m, c_m, wuk, wuv), rtol=1e-4, atol=1e-4)
  sg = np.matmul(qg, np.swapaxes(kg, -1, -2)) * np.float32(0.1767767)
  eg = np.exp(sg - np.max(sg, axis=-1, keepdims=True))
  np.testing.assert_allclose(gqa_out, np.matmul(eg / np.sum(eg, axis=-1, keepdims=True), vg), rtol=1e-4, atol=1e-4)
  ref_aff, ref_fin = _affine_ref_1d(aa, bb, np.float32(0.5))
  np.testing.assert_allclose(aff_out, ref_aff, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(aff_fin, ref_fin, rtol=1e-4, atol=1e-4)
  ref_ssd, ref_s = _mamba2_ssd_ref(qs, ks, vs, as_)
  np.testing.assert_allclose(ssd_out, ref_ssd, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(ssd_s, ref_s, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_simulator_laws34(tmp_path):
  script = tmp_path / "run_laws34_sim.py"
  script.write_text(
    'from tests.test_algebraic_mla_ssd import _run_gpu_laws34\n_run_gpu_laws34("cuda")\nprint("laws34-sim-ok")\n'
  )
  env = dict(os.environ, PARAKEET_CUDA_SIMULATE="1")
  proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, check=False)
  assert proc.returncode == 0 and "laws34-sim-ok" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_gpu_laws34(backend):
  if parakeet.available_backends().get(backend) != "ok":
    pytest.skip(f"{backend} unavailable")
  _run_gpu_laws34(backend)


def test_affine_scan_preserves_order_with_shared_storage_and_repeated_stores():
  @parakeet.jit
  def overlap(a, b, out, s):
    for t in range(a.shape[0]):
      s = a[t] * s + b[t]
      out[t] = s
    return s

  @parakeet.jit
  def scatter(a, b, ix, s):
    out = np.zeros_like(b)
    for t in range(a.shape[0]):
      s = a[t] * s + b[t]
      out[ix[t]] = s
    return out

  a = np.full(257, 0.01, dtype="f")
  b = np.linspace(0, 2, 258, dtype="f")
  idx = np.arange(len(a)) % 7
  with parakeet.options(verify=True, blas=False):
    for backend in ("c", "multicore"):
      original, got = b.copy(), b.copy()
      expected = overlap.__wrapped__(a, original[:-1], original[1:], np.float32(1))
      final = overlap(a, got[:-1], got[1:], np.float32(1), _backend=backend)
      np.testing.assert_allclose(got, original, rtol=2e-6)
      np.testing.assert_allclose(final, expected, rtol=2e-6)
      np.testing.assert_allclose(
        scatter(a, b[:-1], idx, np.float32(1), _backend=backend),
        scatter.__wrapped__(a, b[:-1], idx, np.float32(1)),
        rtol=2e-6,
      )


def test_ssd_preserves_permuted_state_side_effects_and_shape_errors():
  @parakeet.jit
  def permuted(q, k, v, a):
    state = np.zeros((k.shape[1], v.shape[1]), dtype=np.float32)
    out = np.empty((q.shape[0], v.shape[1]), dtype=np.float32)
    for t in range(q.shape[0]):
      state = a[t] * state[:, ::-1] + k[t][:, None] * v[t][None, :]
      out[t] = q[t] @ state
    return out, state

  @parakeet.jit
  def observed(q, k, v, a, seen):
    state = np.zeros((k.shape[1], v.shape[1]), dtype=np.float32)
    out = np.empty((q.shape[0], v.shape[1]), dtype=np.float32)
    for t in range(q.shape[0]):
      state = a[t] * state + k[t][:, None] * v[t][None, :]
      out[t] = q[t] @ state
      seen[t] = 1
    return out, state

  rng = np.random.default_rng(48)
  q, k, v = rng.normal(size=(3, 140, 4)).astype("f")
  a = rng.uniform(0.2, 0.85, len(q)).astype("f")
  with parakeet.options(blas=False, verify=True):
    seen = np.zeros(len(q), dtype="i")
    for f, extra in ((permuted, ()), (observed, (seen,))):
      result = f(q, k, v, a, *extra)
      expected = f.__wrapped__(q, k, v, a, *(x.copy() for x in extra))
      for r, e in zip(result, expected):
        np.testing.assert_allclose(r, e, atol=2e-5)
    np.testing.assert_array_equal(seen, np.ones_like(seen))
    with pytest.raises(ValueError, match="align"):
      _mamba2_ssd(q[:, :2], k, v, a)
