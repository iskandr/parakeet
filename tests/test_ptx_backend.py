"""CPU-only tests for the direct PTX 7.8 emitter, PtxTidy, and PtxCheck verifier."""

from __future__ import annotations

import sys

import numpy as np
import pytest

_SKIP_WIN32_SIM = pytest.mark.skipif(sys.platform == "win32", reason="CUDA simulator requires POSIX C++")

import parakeet as par
from parakeet.backends.cuda import _canonicalize_cuda_kernel
from parakeet.backends.ptx import (
  PtxCheckError,
  PtxEntry,
  PtxInst,
  bundle_ptx_module,
  emit_ptx_entry_body,
  emit_ptx_kernel,
  format_ptx_entry,
  ptx_check,
  ptx_tidy,
  verify_ptx,
)


def _cuda_plans(fn, *args, **opts):
  base = dict(backend="cuda", device_threshold=1, device_fallback="silent", verify=False)
  base.update(opts)
  with par.options(**base):
    spec = fn.specialize(*args)
  return [k.device for k in spec.module.kernels if k.device is not None]


@_SKIP_WIN32_SIM
def test_ptx_plain_elementwise_sqrt_and_where(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.backends.cuda import CudaDevice

  CudaDevice._instance = None
  try:

    @par.jit
    def adam_step(p, g, m, v, lr, b1, b2, bc1, bc2):
      mn = b1 * m + (np.float32(1.0) - b1) * g
      vn = b2 * v + (np.float32(1.0) - b2) * g * g
      pn = p - (lr * mn / bc1) / (np.sqrt(vn / bc2) + np.float32(1e-8))
      return np.where(pn > np.float32(-10.0), pn, np.float32(-10.0)), mn, vn

    x = np.ones(4096, dtype=np.float32)
    s = np.float32(0.1)
    plans = _cuda_plans(adam_step, x, x, x, x, s, s, s, s, s)
    assert plans
    for plan in plans:
      ptx = emit_ptx_kernel(plan)
      assert ptx is not None
      verify_ptx(ptx)
      assert "sqrt.rn.f32" in ptx
      assert "div.rn.f32" in ptx
      assert "selp.f32" in ptx
  finally:
    CudaDevice._instance = None


@_SKIP_WIN32_SIM
def test_ptx_integer_ops_and_bounds_check(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.backends.cuda import CudaDevice

  CudaDevice._instance = None
  try:

    @par.jit
    def int_kernel(a, b, table, idx):
      q = (a // np.maximum(b, np.int64(1))) + (a % np.maximum(b, np.int64(1)))
      bits = (a & b) ^ (a | np.int64(3))
      sh = (a << np.int64(2)) + (b >> np.int64(1))
      return q + bits + sh + table[idx]

    a = np.arange(1, 2049, dtype=np.int64)
    b = np.arange(2, 2050, dtype=np.int64)
    table = np.arange(2048, dtype=np.int64)
    idx = np.arange(2048, dtype=np.int64)
    plans = _cuda_plans(int_kernel, a, b, table, idx, bounds_check=True)
    assert plans
    ptx = emit_ptx_kernel(plans[0])
    assert ptx is not None
    verify_ptx(ptx)
    assert "atom.global.cas.b32" in ptx
    assert "div.s64" in ptx
    assert "rem.s64" in ptx
    assert "xor.b64" in ptx
  finally:
    CudaDevice._instance = None


@_SKIP_WIN32_SIM
def test_ptx_atomic_scatter_and_split_reduce(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.backends.cuda import CudaDevice

  CudaDevice._instance = None
  try:

    @par.jit
    def scatter_hist(bins, vals):
      return par.scatter_add(256, bins, vals)

    bins = np.arange(4096, dtype=np.int64) % 256
    vals = np.ones(4096, dtype=np.float32)
    plans = _cuda_plans(scatter_hist, bins, vals)
    assert plans
    ptx_texts = [emit_ptx_kernel(p) for p in plans]
    assert any(t is not None and "atom.global.add.f32" in t for t in ptx_texts)
    for t in ptx_texts:
      if t is not None:
        verify_ptx(t)
  finally:
    CudaDevice._instance = None


@_SKIP_WIN32_SIM
def test_ptx_warp_row_reduce_layernorm_and_max(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.backends.cuda import CudaDevice

  CudaDevice._instance = None
  try:

    @par.jit
    def layer_norm(x, g, b):
      mu = np.mean(x, axis=1, keepdims=True)
      var = np.mean((x - mu) ** np.float32(2.0), axis=1, keepdims=True)
      return (x - mu) / np.sqrt(var + np.float32(1e-5)) * g + b

    @par.jit
    def row_max_f64(x):
      return np.max(x, axis=1)

    x32 = np.ones((64, 128), dtype=np.float32)
    g32 = np.ones(128, dtype=np.float32)
    b32 = np.zeros(128, dtype=np.float32)
    ln_plans = _cuda_plans(layer_norm, x32, g32, b32)
    assert any(p.loop.warp_reduce is not None for p in ln_plans)
    for p in ln_plans:
      ptx = emit_ptx_kernel(p)
      assert ptx is not None
      verify_ptx(ptx)
      if p.loop.warp_reduce is not None:
        assert "shfl.sync.down.b32" in ptx
        assert "shfl.sync.idx.b32" in ptx
        assert "sqrt.rn.f32" in ptx

    x64 = np.ones((64, 96), dtype=np.float64)
    max_plans = _cuda_plans(row_max_f64, x64)
    assert max_plans
    for p in max_plans:
      ptx = emit_ptx_kernel(p)
      assert ptx is not None
      verify_ptx(ptx)
      if p.loop.warp_reduce is not None:
        assert "shfl.sync.down.b32" in ptx
        assert "mov.b64 {" in ptx
  finally:
    CudaDevice._instance = None


@_SKIP_WIN32_SIM
def test_ptx_fallback_on_libm_and_tiled_gemm(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.backends.cuda import CudaDevice

  CudaDevice._instance = None
  try:

    @par.jit
    def with_tanh(x):
      return np.tanh(x)

    @par.jit
    def matmul_f32(a, b):
      return a @ b

    x = np.ones(4096, dtype=np.float32)
    tanh_plans = _cuda_plans(with_tanh, x)
    assert tanh_plans and emit_ptx_kernel(tanh_plans[0]) is None

    a = np.ones((128, 128), dtype=np.float32)
    mm_plans = _cuda_plans(matmul_f32, a, a)
    assert mm_plans and any(p.loop.shared_tile is not None for p in mm_plans)
    assert all(emit_ptx_kernel(p) is None for p in mm_plans if p.loop.shared_tile is not None)
  finally:
    CudaDevice._instance = None


def test_ptx_tidy_prunes_dead_registers_and_keeps_side_effects():
  entry = PtxEntry(
    name="pk_test",
    params=[(".u64", "p_out")],
    reg_types={"%rd0": ".b64", "%f0": ".f32", "%f1": ".f32", "%f2": ".f32", "%f3": ".f32"},
    insts=[
      PtxInst("ld.param.u64 %rd0, [p_out];", defs=("%rd0",)),
      PtxInst("mov.f32 %f0, 0f3F800000;", defs=("%f0",)),
      PtxInst("add.f32 %f1, %f0, %f0;", defs=("%f1",), uses=("%f0",)),  # dead
      PtxInst("mul.f32 %f2, %f1, %f0;", defs=("%f2",), uses=("%f1", "%f0")),  # dead
      PtxInst("atom.global.add.f32 %f3, [%rd0], %f0;", defs=("%f3",), uses=("%rd0", "%f0"), side_effect=True),
      PtxInst("ret;", side_effect=True),
    ],
  )
  tidied = ptx_tidy(entry)
  text = bundle_ptx_module([format_ptx_entry(tidied)])
  verify_ptx(text)
  assert "%f1" not in tidied.reg_types
  assert "%f2" not in tidied.reg_types
  assert "%f3" in tidied.reg_types  # side-effecting atom destination preserved
  assert len(tidied.insts) == 4


def test_ptx_check_catches_invalid_ptx():
  assert ptx_check(".visible .entry foo() { ret; }")  # missing header
  bad_reg = """
.version 7.8
.target sm_80
.address_size 64
.visible .entry k(.param .u64 p0)
{
  .reg .b64 %rd0;
  ld.param.u64 %rd0, [p0];
  add.s64 %rd1, %rd0, 1;
  ret;
}
"""
  errs = ptx_check(bad_reg)
  assert any("undeclared register %rd1" in e for e in errs)

  bad_op = """
.version 7.8
.target sm_80
.address_size 64
.visible .entry k(.param .s32 p0)
{
  .reg .b32 %r0, %r1;
  ld.param.s32 %r0, [p0];
  mul.s32 %r1, %r0, %r0;
  and.s32 %r1, %r0, 7;
  bra L_missing;
  ret;
}
"""
  errs = ptx_check(bad_op)
  assert any("mul requires .lo" in e for e in errs)
  assert any("bitwise op" in e for e in errs)
  assert any("undefined label L_missing" in e for e in errs)
  with pytest.raises(PtxCheckError):
    verify_ptx(bad_op)


@_SKIP_WIN32_SIM
def test_ptx_nanogpt_coverage_and_bundled_module(monkeypatch):
  monkeypatch.setenv("PARAKEET_CUDA_SIMULATE", "1")
  from parakeet.backends.cuda import CudaDevice

  CudaDevice._instance = None
  try:
    from examples import nanogpt

    rng = np.random.default_rng(0)
    model = nanogpt.init(vocab=65, context=64, width=128, layers=2, rng=rng)
    model = nanogpt.tree_map(lambda a: a.astype(np.float32), model)
    tok = rng.integers(0, 65, size=(16, 64), dtype=np.int64)
    tgt = rng.integers(0, 65, size=(16, 64), dtype=np.int64)
    cas = np.tril(np.ones((64, 64), dtype=bool))

    @par.jit
    def compiled_vg(m, t, y, c):
      return nanogpt.loss_and_grad(m, t, y, c)

    plans = _cuda_plans(compiled_vg, model, tok, tgt, cas, tensor_cores=True)
    unique_by_canon = {}
    for p in plans:
      unique_by_canon.setdefault(_canonicalize_cuda_kernel(p.source), p)

    bodies = []
    for idx, p in enumerate(unique_by_canon.values()):
      body = emit_ptx_entry_body(p, entry_name=f"pk_ptx_{idx}")
      if body is not None:
        bodies.append(body)

    assert len(bodies) >= 0.50 * len(unique_by_canon), (len(bodies), len(unique_by_canon))
    bundled = bundle_ptx_module(bodies, arch="sm_80")
    verify_ptx(bundled)
  finally:
    CudaDevice._instance = None


def _has_real_cuda() -> bool:
  import os

  if os.environ.get("PARAKEET_CUDA_SIMULATE", "").strip().lower() not in ("", "0", "false", "no"):
    return False
  try:
    import cupy

    return cupy.cuda.runtime.getDeviceCount() > 0
  except Exception:  # noqa: BLE001
    return False


def _reset_cuda_backend():
  from parakeet.backends.cuda import CudaDevice

  CudaDevice._instance = None
  par.api._backends.pop("cuda", None)


@pytest.mark.skipif(not _has_real_cuda(), reason="requires physical CUDA GPU")
def test_ptx_real_gpu_differential_parity_and_cu_module_load():
  from parakeet.backends.cuda import CudaDevice
  from parakeet.backends.cuda_rt import try_load_native_cuda_rt

  @par.jit
  def ptx_suite(x, y, bins, a, b):
    # Element-wise + sqrt + where + warp LayerNorm + row max + atomic scatter + integer div/mod
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mu) ** np.float32(2.0), axis=1, keepdims=True)
    ln = (x - mu) / np.sqrt(var + np.float32(1e-5))
    rmax = np.max(x, axis=1)
    hist = par.scatter_add(64, bins, y)
    q = (a // b) + (a % b) + ((a & b) ^ (a | np.int64(5)))
    return ln, rmax, hist, q

  rng = np.random.default_rng(7)
  x = rng.standard_normal((128, 128)).astype(np.float32)
  y = rng.standard_normal(4096).astype(np.float32)
  bins = rng.integers(0, 64, size=4096, dtype=np.int64)
  a = rng.integers(-500, 500, size=4096, dtype=np.int64)
  b = rng.integers(1, 37, size=4096, dtype=np.int64) * rng.choice(np.array([-1, 1], dtype=np.int64), size=4096)

  # Run with direct PTX enabled
  _reset_cuda_backend()
  with par.options(backend="cuda", device_threshold=1, device_fallback="error", verify=False, cuda_compiler="ptx"):
    out_ptx = ptx_suite(x, y, bins, a, b)
    dev_ptx = CudaDevice.get()
    assert dev_ptx.driver.last_ptx_error is None, dev_ptx.driver.last_ptx_error
    assert dev_ptx.driver.ptx_compiled_count >= 3

  # Run with NVRTC only (differential reference)
  _reset_cuda_backend()
  par.api._tables.pop(ptx_suite.fn.__code__, None)
  with par.options(backend="cuda", device_threshold=1, device_fallback="error", verify=False, cuda_compiler="nvrtc"):
    out_nvrtc = ptx_suite(x, y, bins, a, b)

  np.testing.assert_allclose(out_ptx[0], out_nvrtc[0], rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(out_ptx[1], out_nvrtc[1], rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(out_ptx[2], out_nvrtc[2], rtol=1e-4, atol=1e-4)
  np.testing.assert_array_equal(out_ptx[3], out_nvrtc[3])

  # Verify native cuModuleLoadDataEx / cuModuleGetFunction / cuModuleUnload directly
  native = try_load_native_cuda_rt()
  if native is not None:
    minimal_ptx = (".version 7.8\n.target sm_80\n.address_size 64\n.visible .entry pk_noop() {\n  ret;\n}\n").encode(
      "utf-8"
    ) + b"\x00"
    mod_h, funcs = native.load_ptx_module(minimal_ptx, ["pk_noop"])
    assert mod_h and funcs["pk_noop"]
    native.unload_module(mod_h)
