"""Colab NVIDIA L4 benchmark for the Parakeet Algebraic Gallery:
Parakeet (CUDA, 5-15 lines of NumPy/DSL) vs PyTorch eager vs torch.compile vs JAX.

Run on demand::

    PK_COLAB_GPU=L4 runplz colab tools/colab_gallery_bench.py
"""

import os

from runplz import App, Image

GPU = os.environ.get("PK_COLAB_GPU", "L4")
app = App("parakeet-gallery-bench")
image = Image.from_registry("python:3.12")

BENCH_SCRIPT = r"""
import json, time
import numpy as np
import cupy as cp
import torch
import jax
import jax.numpy as jnp
import parakeet
from parakeet import dsl, value_and_grad

def sync_pk():
  cp.cuda.runtime.deviceSynchronize()

def bench_pk(fn, args, warmup=3, iters=20):
  with parakeet.options(backend="cuda", blas=False, device_threshold=1):
    dargs = [parakeet.to_device(a, backend="cuda") if isinstance(a, np.ndarray) else a for a in args]
    t0 = time.perf_counter()
    out = fn(*dargs)
    sync_pk()
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
      out = fn(*dargs)
    sync_pk()
    ts = []
    for _ in range(iters):
      t1 = time.perf_counter()
      out = fn(*dargs)
      sync_pk()
      ts.append((time.perf_counter() - t1) * 1e3)
    return {"compile_s": round(compile_s, 2), "median_ms": round(float(np.median(ts)), 3)}, out

def bench_torch(fn, args, compile_mode=False, warmup=5, iters=20):
  targs = [torch.from_numpy(a).cuda() if isinstance(a, np.ndarray) else a for a in args]
  f = torch.compile(fn) if compile_mode else fn
  t0 = time.perf_counter()
  out = f(*targs)
  torch.cuda.synchronize()
  compile_s = time.perf_counter() - t0
  for _ in range(warmup):
    out = f(*targs)
  torch.cuda.synchronize()
  ts = []
  for _ in range(iters):
    t1 = time.perf_counter()
    out = f(*targs)
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t1) * 1e3)
  return {"compile_s": round(compile_s, 2), "median_ms": round(float(np.median(ts)), 3)}, out

def bench_jax(fn, args, warmup=5, iters=20):
  jargs = [jnp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
  f = jax.jit(fn)
  t0 = time.perf_counter()
  out = f(*jargs)
  jax.block_until_ready(out)
  compile_s = time.perf_counter() - t0
  for _ in range(warmup):
    out = f(*jargs)
  jax.block_until_ready(out)
  ts = []
  for _ in range(iters):
    t1 = time.perf_counter()
    out = f(*jargs)
    jax.block_until_ready(out)
    ts.append((time.perf_counter() - t1) * 1e3)
  return {"compile_s": round(compile_s, 2), "median_ms": round(float(np.median(ts)), 3)}, out

# --- Kernels ---
@dsl.kernel
def pk_rbf(xq, xt, yt, ng):
  d2 = np.sum((xq[:, None, :] - xt[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * ng
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ yt

def th_rbf(xq, xt, yt, ng):
  d2 = torch.cdist(xq, xt, compute_mode="donot_use_mm_for_euclid_dist") ** 2
  p = torch.softmax(d2 * ng, dim=-1)
  return p @ yt

def jx_rbf(xq, xt, yt, ng):
  d2 = jnp.sum((xq[:, None, :] - xt[None, :, :]) ** 2, axis=-1)
  p = jax.nn.softmax(d2 * ng, axis=-1)
  return p @ yt

@dsl.kernel
def pk_flash_causal(q, k, v):
  T = q.shape[-2]
  mask = np.arange(T)[:, None] >= np.arange(T)[None, :]
  s = np.where(mask, np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(0.125), np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)

def th_flash_naive(q, k, v):
  T = q.shape[-2]
  mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=q.device))
  s = torch.where(mask, torch.matmul(q, k.transpose(-1, -2)) * 0.125, -1e9)
  return torch.matmul(torch.softmax(s, dim=-1), v)

def jx_flash_naive(q, k, v):
  T = q.shape[-2]
  mask = jnp.tril(jnp.ones((T, T), dtype=bool))
  s = jnp.where(mask, jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * 0.125, -1e9)
  return jnp.matmul(jax.nn.softmax(s, axis=-1), v)

@dsl.kernel
def pk_mla(q, c, wuk, wuv):
  k = c @ wuk
  v = c @ wuv
  s = (q @ k.T) * np.float32(0.0625)
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v

def th_mla(q, c, wuk, wuv):
  k = c @ wuk
  v = c @ wuv
  s = (q @ k.T) * 0.0625
  return torch.softmax(s, dim=-1) @ v

def jx_mla(q, c, wuk, wuv):
  k = c @ wuk
  v = c @ wuv
  s = (q @ k.T) * 0.0625
  return jax.nn.softmax(s, axis=-1) @ v

@dsl.kernel
def pk_tropical(a, b):
  m, n = a.shape[0], b.shape[1]
  k = a.shape[1]
  return dsl.imap(
    lambda ij: dsl.ireduce(lambda r: a[ij[0], r] + b[r, ij[1]], k, combine=parakeet.minimum),
    (m, n),
  )

def th_tropical(a, b):
  return torch.min(a[:, :, None] + b[None, :, :], dim=1).values

def jx_tropical(a, b):
  return jnp.min(a[:, :, None] + b[None, :, :], axis=1)

@dsl.kernel
def pk_1nn(xq, xr):
  return dsl.map(lambda q: np.argmin(dsl.map(lambda r: np.sum((q - r) ** np.float32(2.0)), xr)), xq)

def th_1nn(xq, xr):
  return torch.argmin(torch.cdist(xq, xr), dim=-1)

def jx_1nn(xq, xr):
  return jnp.argmin(jnp.sum((xq[:, None, :] - xr[None, :, :]) ** 2, axis=-1), axis=-1)

@dsl.kernel
def pk_cce(h, w, tgt):
  logits = h @ w
  m = np.max(logits, axis=-1, keepdims=True)
  z = logits - m
  lse = np.log(np.sum(np.exp(z), axis=-1))
  picked = z[np.arange(h.shape[0]), tgt]
  return -np.mean(picked - lse)

def th_cce(h, w, tgt):
  return torch.nn.functional.cross_entropy(h @ w, tgt)

def jx_cce(h, w, tgt):
  logits = h @ w
  lse = jax.nn.logsumexp(logits, axis=-1)
  picked = logits[jnp.arange(h.shape[0]), tgt]
  return -jnp.mean(picked - lse)

def main():
  rng = np.random.default_rng(0)
  out = {}

  # 1. RBF Kernel Regression (4096 x 8192 x 32 -> 4096*8192 = 33M pairs; naive OOMs or thrashes HBM!)
  xq = rng.standard_normal((2048, 32)).astype(np.float32)
  xt = rng.standard_normal((4096, 32)).astype(np.float32)
  yt = rng.standard_normal((4096, 16)).astype(np.float32)
  ng = np.float32(-0.05)
  row = {}
  row["parakeet"], _ = bench_pk(pk_rbf, (xq, xt, yt, ng))
  row["torch_eager"], _ = bench_torch(th_rbf, (xq, xt, yt, -0.05), False)
  row["torch_compile"], _ = bench_torch(th_rbf, (xq, xt, yt, -0.05), True)
  row["jax_jit"], _ = bench_jax(jx_rbf, (xq, xt, yt, -0.05))
  out["rbf_2048x4096x32"] = row
  print("RBF:", row, flush=True)

  # 2. Causal FlashAttention (B=4, H=8, T=1024, D=64)
  q = rng.standard_normal((4, 8, 1024, 64)).astype(np.float32) * 0.2
  k = rng.standard_normal((4, 8, 1024, 64)).astype(np.float32) * 0.2
  v = rng.standard_normal((4, 8, 1024, 64)).astype(np.float32) * 0.2
  row = {}
  row["parakeet"], _ = bench_pk(pk_flash_causal, (q, k, v))
  row["torch_eager_naive"], _ = bench_torch(th_flash_naive, (q, k, v), False)
  row["torch_compile_naive"], _ = bench_torch(th_flash_naive, (q, k, v), True)
  row["jax_jit_naive"], _ = bench_jax(jx_flash_naive, (q, k, v))
  out["causal_attn_B4_H8_T1024_D64"] = row
  print("FLASH:", row, flush=True)

  # 3. Cut Cross-Entropy fwd (M=1024, K=256, V=32768 -> 33M logits!)
  h = rng.standard_normal((1024, 256)).astype(np.float32) * 0.1
  w = rng.standard_normal((256, 32768)).astype(np.float32) * 0.1
  tgt = rng.integers(0, 32768, size=(1024,), dtype=np.int64)
  row = {}
  row["parakeet"], _ = bench_pk(pk_cce, (h, w, tgt))
  row["torch_eager"], _ = bench_torch(th_cce, (h, w, tgt), False)
  row["torch_compile"], _ = bench_torch(th_cce, (h, w, tgt), True)
  row["jax_jit"], _ = bench_jax(jx_cce, (h, w, tgt))
  out["cce_M1024_K256_V32768"] = row
  print("CCE:", row, flush=True)

  # 4. DeepSeek MLA Decode (Tq=4, Tkv=8192, dc=64, dh=256)
  qm = rng.standard_normal((4, 256)).astype(np.float32) * 0.1
  cm = rng.standard_normal((8192, 64)).astype(np.float32) * 0.1
  wuk = rng.standard_normal((64, 256)).astype(np.float32) * 0.1
  wuv = rng.standard_normal((64, 256)).astype(np.float32) * 0.1
  row = {}
  row["parakeet"], _ = bench_pk(pk_mla, (qm, cm, wuk, wuv))
  row["torch_eager"], _ = bench_torch(th_mla, (qm, cm, wuk, wuv), False)
  row["torch_compile"], _ = bench_torch(th_mla, (qm, cm, wuk, wuv), True)
  row["jax_jit"], _ = bench_jax(jx_mla, (qm, cm, wuk, wuv))
  out["mla_decode_Tq4_Tkv8192_dc64_dh256"] = row
  print("MLA:", row, flush=True)

  # 5. Tropical Min-Plus APSP (512 x 512)
  ta = rng.uniform(0.1, 5.0, size=(512, 512)).astype(np.float32)
  tb = rng.uniform(0.1, 5.0, size=(512, 512)).astype(np.float32)
  row = {}
  row["parakeet"], _ = bench_pk(pk_tropical, (ta, tb))
  row["torch_eager"], _ = bench_torch(th_tropical, (ta, tb), False)
  row["torch_compile"], _ = bench_torch(th_tropical, (ta, tb), True)
  row["jax_jit"], _ = bench_jax(jx_tropical, (ta, tb))
  out["tropical_apsp_512"] = row
  print("TROPICAL:", row, flush=True)

  # 6. Exact 1-NN (2048 x 4096 x 32)
  row = {}
  row["parakeet"], _ = bench_pk(pk_1nn, (xq, xt))
  row["torch_eager"], _ = bench_torch(th_1nn, (xq, xt), False)
  row["torch_compile"], _ = bench_torch(th_1nn, (xq, xt), True)
  row["jax_jit"], _ = bench_jax(jx_1nn, (xq, xt))
  out["exact_1nn_2048x4096x32"] = row
  print("1NN:", row, flush=True)

  print("\nGALLERY_JSON:" + json.dumps(out), flush=True)

if __name__ == "__main__":
  main()
"""


@app.function(image=image, gpu=GPU, timeout=20 * 60)
def run_bench():
  import pathlib
  import subprocess
  import sys

  repo = pathlib.Path(__file__).resolve().parents[1]
  py = sys.executable
  subprocess.run(["bash", "-c", "nvidia-smi; nproc; python --version"], check=False)
  subprocess.run([py, "-m", "pip", "install", "-q", "-e", "."], cwd=repo, check=True)
  script = repo / "_gallery_bench.py"
  script.write_text(BENCH_SCRIPT)
  env = dict(os.environ, PYTHONPATH=str(repo))
  subprocess.run([py, str(script)], cwd=repo, env=env, check=True)


@app.local_entrypoint()
def main():
  run_bench.remote()
