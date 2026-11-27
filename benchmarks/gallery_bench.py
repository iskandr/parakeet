"""Cross-framework GPU benchmark for the kernels in ``examples/algebraic_gallery.py``.

Imports the Parakeet kernels directly from ``examples.algebraic_gallery`` so the
benchmarked code is the exact documented source, and compares them against
PyTorch (eager and ``torch.compile``), JAX (``jax.jit``), and CuPy.

Run directly on a CUDA machine::

    python -m benchmarks.gallery_bench
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import parakeet as par
from examples.algebraic_gallery import (
  causal_flash_attention,
  cut_cross_entropy,
  deepseek_mla,
  exact_1nn,
  gbdt_best_split,
  gbdt_best_split_comprehension,
  gbdt_best_split_loop,
  gbdt_best_split_numpy,
  nearest_fold,
  nearest_numpy,
  rbf_kernel_regression,
  tropical_min_plus,
)


def _time_gpu(call, sync, *, warmup: int = 3, iters: int = 20) -> tuple[dict, object]:
  sync()
  t0 = time.perf_counter()
  out = call()
  sync()
  compile_s = time.perf_counter() - t0

  for _ in range(warmup):
    out = call()
  sync()

  samples_ms = []
  for _ in range(iters):
    t1 = time.perf_counter()
    out = call()
    sync()
    samples_ms.append((time.perf_counter() - t1) * 1e3)

  arr = np.asarray(samples_ms, dtype=np.float64)
  q25, q75 = np.percentile(arr, [25, 75])
  return {
    "compile_s": round(compile_s, 3),
    "median_ms": round(float(statistics.median(samples_ms)), 3),
    "iqr_ms": round(float(q75 - q25), 3),
  }, out


def bench_parakeet(fn, args, *, tensor_cores: bool = False, iters: int = 20):
  import cupy as cp

  sync = cp.cuda.runtime.deviceSynchronize
  with par.options(backend="cuda", blas=False, device_threshold=1, tensor_cores=tensor_cores):
    dargs = [par.to_device(a, backend="cuda") if isinstance(a, np.ndarray) else a for a in args]
    stats, out = _time_gpu(lambda: fn(*dargs), sync, iters=iters)
    return stats, par.to_host(out)


def bench_torch(fn, args, *, compiled: bool = False, tf32: bool = False, iters: int = 20):
  import torch

  torch.backends.cuda.matmul.allow_tf32 = tf32
  torch.set_float32_matmul_precision("high" if tf32 else "highest")
  targs = [torch.from_numpy(a).cuda() if isinstance(a, np.ndarray) else a for a in args]
  runner = torch.compile(fn) if compiled else fn
  stats, out = _time_gpu(lambda: runner(*targs), torch.cuda.synchronize, warmup=5, iters=iters)
  return stats, out.detach().cpu().numpy()


def bench_jax(fn, args, *, tf32: bool = False, iters: int = 20):
  import jax
  import jax.numpy as jnp

  jax.config.update("jax_default_matmul_precision", "tensorfloat32" if tf32 else "highest")
  jargs = [jnp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
  compiled = jax.jit(fn)
  last = [None]

  def call():
    last[0] = compiled(*jargs)
    return last[0]

  def sync():
    if last[0] is not None:
      jax.block_until_ready(last[0])

  stats, out = _time_gpu(call, sync, warmup=5, iters=iters)
  return stats, np.asarray(out)


def _check_close(name: str, got, ref, *, exact: bool = False, tol: float = 2e-3) -> None:
  if exact:
    np.testing.assert_array_equal(got, ref, err_msg=name)
  else:
    np.testing.assert_allclose(got, ref, rtol=tol, atol=tol, err_msg=name)


def run_all(*, iters: int = 20) -> dict:
  import jax
  import jax.numpy as jnp
  import torch

  rng = np.random.default_rng(0)
  results = {}

  # 1. RBF kernel regression
  xq = rng.standard_normal((2048, 32)).astype(np.float32)
  xt = rng.standard_normal((4096, 32)).astype(np.float32)
  yt = rng.standard_normal((4096, 16)).astype(np.float32)
  ng = np.float32(-0.05)

  def th_rbf(q, t, y, gamma):
    d2 = torch.cdist(q, t, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    return torch.softmax(d2 * gamma, dim=-1) @ y

  def jx_rbf(q, t, y, gamma):
    d2 = jnp.sum((q[:, None, :] - t[None, :, :]) ** 2, axis=-1)
    return jax.nn.softmax(d2 * gamma, axis=-1) @ y

  row = {}
  row["parakeet_fp32"], ref_rbf = bench_parakeet(rbf_kernel_regression, (xq, xt, yt, ng), iters=iters)
  row["torch_eager"], got = bench_torch(th_rbf, (xq, xt, yt, -0.05), compiled=False, iters=iters)
  _check_close("rbf/torch_eager", got, ref_rbf)
  row["torch_compile"], got = bench_torch(th_rbf, (xq, xt, yt, -0.05), compiled=True, iters=iters)
  _check_close("rbf/torch_compile", got, ref_rbf)
  row["jax_jit"], got = bench_jax(jx_rbf, (xq, xt, yt, -0.05), iters=iters)
  _check_close("rbf/jax_jit", got, ref_rbf)
  results["rbf_2048x4096x32"] = row
  print("RBF:", row, flush=True)

  # 2. Causal attention
  q = rng.standard_normal((4, 8, 1024, 64)).astype(np.float32) * 0.2
  k = rng.standard_normal((4, 8, 1024, 64)).astype(np.float32) * 0.2
  v = rng.standard_normal((4, 8, 1024, 64)).astype(np.float32) * 0.2

  def th_attn_explicit(q_, k_, v_):
    seq = q_.shape[-2]
    mask = torch.tril(torch.ones((seq, seq), dtype=torch.bool, device=q_.device))
    scores = torch.where(mask, torch.matmul(q_, k_.transpose(-1, -2)) * 0.125, -1e9)
    return torch.matmul(torch.softmax(scores, dim=-1), v_)

  def th_attn_sdpa(q_, k_, v_):
    return torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=True)

  def jx_attn_explicit(q_, k_, v_):
    seq = q_.shape[-2]
    mask = jnp.tril(jnp.ones((seq, seq), dtype=bool))
    scores = jnp.where(mask, jnp.matmul(q_, jnp.swapaxes(k_, -1, -2)) * 0.125, -1e9)
    return jnp.matmul(jax.nn.softmax(scores, axis=-1), v_)

  row = {}
  row["parakeet_fp32"], ref_attn = bench_parakeet(causal_flash_attention, (q, k, v), tensor_cores=False, iters=iters)
  row["parakeet_tf32"], got = bench_parakeet(causal_flash_attention, (q, k, v), tensor_cores=True, iters=iters)
  _check_close("attn/parakeet_tf32", got, ref_attn, tol=5e-3)
  row["torch_eager_explicit"], got = bench_torch(th_attn_explicit, (q, k, v), compiled=False, iters=iters)
  _check_close("attn/torch_eager", got, ref_attn)
  row["torch_compile_explicit"], got = bench_torch(th_attn_explicit, (q, k, v), compiled=True, iters=iters)
  _check_close("attn/torch_compile", got, ref_attn)
  row["torch_sdpa"], got = bench_torch(th_attn_sdpa, (q, k, v), compiled=False, iters=iters)
  _check_close("attn/torch_sdpa", got, ref_attn)
  row["jax_jit_explicit"], got = bench_jax(jx_attn_explicit, (q, k, v), iters=iters)
  _check_close("attn/jax_jit", got, ref_attn)
  results["causal_attn_B4_H8_T1024_D64"] = row
  print("CAUSAL_ATTN:", row, flush=True)

  # 3. Linear cross-entropy (cut cross-entropy)
  h = rng.standard_normal((1024, 256)).astype(np.float32) * 0.1
  w = rng.standard_normal((256, 32768)).astype(np.float32) * 0.1
  tgt = rng.integers(0, 32768, size=(1024,), dtype=np.int64)
  pk_cce_jit = par.jit(cut_cross_entropy)

  def th_cce(h_, w_, tgt_):
    return torch.nn.functional.cross_entropy(h_ @ w_, tgt_)

  def jx_cce(h_, w_, tgt_):
    logits = h_ @ w_
    lse = jax.nn.logsumexp(logits, axis=-1)
    picked = logits[jnp.arange(h_.shape[0]), tgt_]
    return -jnp.mean(picked - lse)

  row = {}
  row["parakeet_fp32"], ref_cce = bench_parakeet(pk_cce_jit, (h, w, tgt), tensor_cores=False, iters=iters)
  row["parakeet_tf32"], got = bench_parakeet(pk_cce_jit, (h, w, tgt), tensor_cores=True, iters=iters)
  _check_close("cce/parakeet_tf32", got, ref_cce, tol=5e-3)
  row["torch_eager"], got = bench_torch(th_cce, (h, w, tgt), compiled=False, iters=iters)
  _check_close("cce/torch_eager", got, ref_cce)
  row["torch_compile"], got = bench_torch(th_cce, (h, w, tgt), compiled=True, iters=iters)
  _check_close("cce/torch_compile", got, ref_cce)
  row["jax_jit"], got = bench_jax(jx_cce, (h, w, tgt), iters=iters)
  _check_close("cce/jax_jit", got, ref_cce)
  results["cce_M1024_K256_V32768"] = row
  print("CCE:", row, flush=True)

  # 4. Multi-head latent attention (MLA decode)
  qm = rng.standard_normal((4, 256)).astype(np.float32) * 0.1
  cm = rng.standard_normal((8192, 64)).astype(np.float32) * 0.1
  wuk = rng.standard_normal((64, 256)).astype(np.float32) * 0.1
  wuv = rng.standard_normal((64, 256)).astype(np.float32) * 0.1

  def th_mla(q_, c_, k_proj, v_proj):
    s = (q_ @ (c_ @ k_proj).T) * 0.0625
    return torch.softmax(s, dim=-1) @ (c_ @ v_proj)

  def jx_mla(q_, c_, k_proj, v_proj):
    s = (q_ @ (c_ @ k_proj).T) * 0.0625
    return jax.nn.softmax(s, axis=-1) @ (c_ @ v_proj)

  row = {}
  row["parakeet_fp32"], ref_mla = bench_parakeet(deepseek_mla, (qm, cm, wuk, wuv), iters=iters)
  row["torch_eager"], got = bench_torch(th_mla, (qm, cm, wuk, wuv), compiled=False, iters=iters)
  _check_close("mla/torch_eager", got, ref_mla)
  row["torch_compile"], got = bench_torch(th_mla, (qm, cm, wuk, wuv), compiled=True, iters=iters)
  _check_close("mla/torch_compile", got, ref_mla)
  row["jax_jit"], got = bench_jax(jx_mla, (qm, cm, wuk, wuv), iters=iters)
  _check_close("mla/jax_jit", got, ref_mla)
  results["mla_decode_Tq4_Tkv8192_dc64_dh256"] = row
  print("MLA:", row, flush=True)

  # 5. Tropical min-plus matrix product
  ta = rng.uniform(0.1, 5.0, size=(512, 512)).astype(np.float32)
  tb = rng.uniform(0.1, 5.0, size=(512, 512)).astype(np.float32)

  def th_tropical(a, b):
    return torch.min(a[:, :, None] + b[None, :, :], dim=1).values

  def jx_tropical(a, b):
    return jnp.min(a[:, :, None] + b[None, :, :], axis=1)

  row = {}
  row["parakeet_fp32"], ref_trop = bench_parakeet(tropical_min_plus, (ta, tb), iters=iters)
  row["torch_eager"], got = bench_torch(th_tropical, (ta, tb), compiled=False, iters=iters)
  _check_close("tropical/torch_eager", got, ref_trop)
  row["torch_compile"], got = bench_torch(th_tropical, (ta, tb), compiled=True, iters=iters)
  _check_close("tropical/torch_compile", got, ref_trop)
  row["jax_jit"], got = bench_jax(jx_tropical, (ta, tb), iters=iters)
  _check_close("tropical/jax_jit", got, ref_trop)
  results["tropical_min_plus_512"] = row
  print("TROPICAL:", row, flush=True)

  # 6. Exact 1-NN (comparing Parakeet styles: broadcast NumPy, nested map, explicit fold)
  def th_1nn(q_, r_):
    return torch.argmin(torch.cdist(q_, r_), dim=-1)

  def jx_1nn(q_, r_):
    return jnp.argmin(jnp.sum((q_[:, None, :] - r_[None, :, :]) ** 2, axis=-1), axis=-1)

  row = {}
  row["parakeet_numpy"], ref_1nn = bench_parakeet(nearest_numpy, (xq, xt), iters=iters)
  row["parakeet_map"], got = bench_parakeet(exact_1nn, (xq, xt), iters=iters)
  _check_close("1nn/parakeet_map", got, ref_1nn, exact=True)
  row["parakeet_fold"], got = bench_parakeet(nearest_fold, (xq, xt), iters=iters)
  _check_close("1nn/parakeet_fold", got, ref_1nn, exact=True)
  row["torch_eager"], got = bench_torch(th_1nn, (xq, xt), compiled=False, iters=iters)
  _check_close("1nn/torch_eager", got, ref_1nn, exact=True)
  row["torch_compile"], got = bench_torch(th_1nn, (xq, xt), compiled=True, iters=iters)
  _check_close("1nn/torch_compile", got, ref_1nn, exact=True)
  row["jax_jit"], got = bench_jax(jx_1nn, (xq, xt), iters=iters)
  _check_close("1nn/jax_jit", got, ref_1nn, exact=True)
  results["exact_1nn_2048x4096x32"] = row
  print("1NN:", row, flush=True)

  # 7. GBDT histogram split (comparing all four Parakeet surface forms)
  bins = rng.integers(0, 256, size=(2_000_000,), dtype=np.int64)
  grads = rng.standard_normal((2_000_000,)).astype(np.float32) + np.where(bins < 71, -0.25, 0.25).astype(np.float32)
  hess = rng.uniform(0.1, 1.0, size=(2_000_000,)).astype(np.float32)
  lam = np.float32(1.0)
  row = {}
  for label, fn in (
    ("parakeet_scatter", gbdt_best_split),
    ("parakeet_comprehension", gbdt_best_split_comprehension),
    ("parakeet_numpy_at", gbdt_best_split_numpy),
    ("parakeet_loop", gbdt_best_split_loop),
  ):
    row[label], _ = bench_parakeet(fn, (bins, grads, hess, 256, lam), iters=iters)
  results["gbdt_split_2m_256"] = row
  print("GBDT_SPLIT:", row, flush=True)

  return results


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args(argv)

  out = run_all(iters=args.iters)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
  print("\nGALLERY_JSON:" + json.dumps(out), flush=True)


if __name__ == "__main__":
  main()
