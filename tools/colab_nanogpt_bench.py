"""Colab GPU benchmark: Parakeet vs PyTorch (eager + torch.compile) vs JAX (jax.jit) on batched float32 nanoGPT.

Run on demand with ``runplz``::

    PK_COLAB_GPU=L4 runplz colab tools/colab_nanogpt_bench.py
    PK_COLAB_GPU=T4 runplz colab tools/colab_nanogpt_bench.py
"""

import os

from runplz import App, Image

GPU = os.environ.get("PK_COLAB_GPU", "L4")
app = App("parakeet-nanogpt-bench")
image = Image.from_registry("python:3.12")

BENCH_SCRIPT = r"""
import json, os, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# Shared GPT initialization (identical float32 weights across all frameworks)
# ---------------------------------------------------------------------------

def init_gpt(vocab, context, width, layers, seed=0):
  rng = np.random.default_rng(seed)
  def w(*shape):
    return rng.normal(0.0, 0.02, shape).astype(np.float32)
  def one_block():
    ones = np.ones(width, dtype=np.float32)
    zeros = np.zeros(width, dtype=np.float32)
    return (
      ones, zeros, w(width, width), w(width, width), w(width, width), w(width, width),
      ones.copy(), zeros.copy(), w(width, 4 * width), np.zeros(4 * width, dtype=np.float32),
      w(4 * width, width), zeros.copy(),
    )
  blocks = tuple(one_block() for _ in range(layers))
  return (w(vocab, width), w(context, width), blocks, np.ones(width, dtype=np.float32), np.zeros(width, dtype=np.float32), w(width, vocab))

def tree_zeros(t):
  return tuple(tree_zeros(x) for x in t) if isinstance(t, tuple) else np.zeros_like(t)

def tree_copy(t):
  return tuple(tree_copy(x) for x in t) if isinstance(t, tuple) else t.copy()

# ---------------------------------------------------------------------------
# 1. Parakeet (CUDA + DeviceArray residency + in-JIT Adam)
# ---------------------------------------------------------------------------
import parakeet
from parakeet import jit, value_and_grad

def pk_layer_norm(x, g, b):
  mu = np.mean(x, axis=2, keepdims=True)
  var = np.mean((x - mu) ** np.float32(2.0), axis=2, keepdims=True)
  return (x - mu) / np.sqrt(var + np.float32(1e-5)) * g + b

def pk_softmax_last(x):
  m = np.max(x, axis=3, keepdims=True)
  e = np.exp(x - m)
  return e / np.sum(e, axis=3, keepdims=True)

def pk_attention(x, wq, wk, wv, wo, causal, n_head):
  B, T, C = x.shape
  D = C // n_head
  q = (x @ wq).reshape((B, T, n_head, D)).transpose((0, 2, 1, 3))
  k = (x @ wk).reshape((B, T, n_head, D)).transpose((0, 2, 1, 3))
  v = (x @ wv).reshape((B, T, n_head, D)).transpose((0, 2, 1, 3))
  scores = (q @ k.transpose((0, 1, 3, 2))) * np.float32(1.0 / np.sqrt(D))
  weights = pk_softmax_last(np.where(causal, scores, np.float32(-1e9)))
  out = (weights @ v).transpose((0, 2, 1, 3)).reshape((B, T, C))
  return out @ wo

def pk_block(x, p, causal, n_head):
  g1, b1, wq, wk, wv, wo, g2, b2, w1, c1, w2, c2 = p
  x = x + pk_attention(pk_layer_norm(x, g1, b1), wq, wk, wv, wo, causal, n_head)
  h = np.tanh(pk_layer_norm(x, g2, b2) @ w1 + c1)
  return x + h @ w2 + c2

def pk_gpt_loss(params, tokens, targets, causal, n_head):
  wte, wpe, blocks, gf, bf, head = params
  B, T = tokens.shape
  x = wte[tokens.reshape((B * T,))].reshape((B, T, wte.shape[1])) + wpe
  for p in blocks:
    x = pk_block(x, p, causal, n_head)
  logits = pk_layer_norm(x, gf, bf) @ head
  flat = logits.reshape((B * T, logits.shape[2]))
  m = np.max(flat, axis=1, keepdims=True)
  z = flat - m
  lse = np.log(np.sum(np.exp(z), axis=1))
  picked = z[np.arange(B * T), targets.reshape((B * T,))]
  return -np.mean(picked - lse)

@jit
def pk_adam_step(params, grads, m, v, lr, b1, b2, bc1, bc2):
  def upd(p, g, mi, vi):
    mn = b1 * mi + (np.float32(1.0) - b1) * g
    vn = b2 * vi + (np.float32(1.0) - b2) * g * g
    pn = p - (lr * mn / bc1) / (np.sqrt(vn / bc2) + np.float32(1e-8))
    return pn, mn, vn
  wte, wpe, blocks, gf, bf, head = params
  gwte, gwpe, gblocks, ggf, gbf, ghead = grads
  mwte, mwpe, mblocks, mgf, mbf, mhead = m
  vwte, vwpe, vblocks, vgf, vbf, vhead = v
  wte, mwte, vwte = upd(wte, gwte, mwte, vwte)
  wpe, mwpe, vwpe = upd(wpe, gwpe, mwpe, vwpe)
  gf, mgf, vgf = upd(gf, ggf, mgf, vgf)
  bf, mbf, vbf = upd(bf, gbf, mbf, vbf)
  head, mhead, vhead = upd(head, ghead, mhead, vhead)
  nblocks = (); nmblocks = (); nvblocks = ()
  for pb, gb, mb, vb in zip(blocks, gblocks, mblocks, vblocks):
    npb = (); nmb = (); nvb = ()
    for pi, gi, mi, vi in zip(pb, gb, mb, vb):
      pn, mn, vn = upd(pi, gi, mi, vi)
      npb = npb + (pn,); nmb = nmb + (mn,); nvb = nvb + (vn,)
    nblocks = nblocks + (npb,); nmblocks = nmblocks + (nmb,); nvblocks = nvblocks + (nvb,)
  return (wte, wpe, nblocks, gf, bf, head), (mwte, mwpe, nmblocks, mgf, mbf, mhead), (vwte, vwpe, nvblocks, vgf, vbf, vhead)

def bench_parakeet(cfg, params_np, tokens_np, targets_np, causal_np, tensor_cores=False, steps=20):
  import cupy
  B, T, C, H, L, V = cfg
  vg = value_and_grad(pk_gpt_loss)
  lr, b1, b2 = np.float32(3e-3), np.float32(0.9), np.float32(0.99)
  with parakeet.options(backend="cuda", device_threshold=1, device_fallback="silent", verify=False, tensor_cores=tensor_cores):
    params = parakeet.to_device(tree_copy(params_np), backend="cuda")
    m = parakeet.to_device(tree_zeros(params_np), backend="cuda")
    v = parakeet.to_device(tree_zeros(params_np), backend="cuda")
    tok_d = parakeet.to_device(tokens_np, backend="cuda")
    tgt_d = parakeet.to_device(targets_np, backend="cuda")
    cas_d = parakeet.to_device(causal_np, backend="cuda")

    t_c0 = time.perf_counter()
    loss0, grads = vg(params, tok_d, tgt_d, cas_d, H)
    bc1, bc2 = np.float32(1.0 - 0.9), np.float32(1.0 - 0.99)
    params, m, v = pk_adam_step(params, grads, m, v, lr, b1, b2, bc1, bc2)
    cupy.cuda.Stream.null.synchronize()
    compile_s = time.perf_counter() - t_c0

    # 2 extra warmup steps
    for s in (2, 3):
      lv, grads = vg(params, tok_d, tgt_d, cas_d, H)
      params, m, v = pk_adam_step(params, grads, m, v, lr, b1, b2, np.float32(1.0 - 0.9**s), np.float32(1.0 - 0.99**s))
    cupy.cuda.Stream.null.synchronize()

    dev = parakeet.api.get_backend("cuda").device()
    u0, d0, l0 = dev.uploads, dev.downloads, dev.launches
    times = []
    for s in range(4, 4 + steps):
      t0 = time.perf_counter()
      lv, grads = vg(params, tok_d, tgt_d, cas_d, H)
      params, m, v = pk_adam_step(params, grads, m, v, lr, b1, b2, np.float32(1.0 - 0.9**s), np.float32(1.0 - 0.99**s))
      cupy.cuda.Stream.null.synchronize()
      times.append((time.perf_counter() - t0) * 1000.0)
    return {
      "compile_s": round(compile_s, 3),
      "median_ms": round(float(np.median(times)), 3),
      "loss0": round(float(loss0), 5),
      "loss_end": round(float(lv), 5),
      "uploads_per_step": round((dev.uploads - u0) / steps, 2),
      "downloads_per_step": round((dev.downloads - d0) / steps, 2),
      "launches_per_step": round((dev.launches - l0) / steps, 1),
    }

# ---------------------------------------------------------------------------
# 2. PyTorch (eager + torch.compile)
# ---------------------------------------------------------------------------
def bench_torch(cfg, params_np, tokens_np, targets_np, causal_np, compiled=False, steps=20):
  import torch
  import torch.nn.functional as F
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  device = torch.device("cuda")
  B, T, C, H, L, V = cfg

  def to_t(x, req=True):
    if isinstance(x, tuple):
      return tuple(to_t(e, req) for e in x)
    return torch.tensor(x, device=device, dtype=torch.float32, requires_grad=req)

  def flat_params(t, out):
    if isinstance(t, tuple):
      for e in t: flat_params(e, out)
    else:
      out.append(t)

  params = to_t(params_np, True)
  plist = []
  flat_params(params, plist)
  tok = torch.tensor(tokens_np, device=device, dtype=torch.long)
  tgt = torch.tensor(targets_np, device=device, dtype=torch.long)
  causal = torch.tensor(causal_np, device=device, dtype=torch.bool)
  opt = torch.optim.Adam(plist, lr=3e-3, betas=(0.9, 0.99), eps=1e-8)

  def ln(x, g, b):
    mu = x.mean(dim=2, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=2, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * g + b

  def fwd_loss():
    wte, wpe, blocks, gf, bf, head = params
    x = wte[tok] + wpe
    D = C // H
    scale = 1.0 / (D ** 0.5)
    for g1, b1, wq, wk, wv, wo, g2, b2, w1, c1, w2, c2 in blocks:
      nx = ln(x, g1, b1)
      q = (nx @ wq).view(B, T, H, D).transpose(1, 2)
      k = (nx @ wk).view(B, T, H, D).transpose(1, 2)
      v = (nx @ wv).view(B, T, H, D).transpose(1, 2)
      scores = (q @ k.transpose(-2, -1)) * scale
      scores = torch.where(causal, scores, torch.tensor(-1e9, device=device, dtype=torch.float32))
      weights = torch.softmax(scores, dim=-1)
      attn = (weights @ v).transpose(1, 2).contiguous().view(B, T, C) @ wo
      x = x + attn
      h = torch.tanh(ln(x, g2, b2) @ w1 + c1)
      x = x + h @ w2 + c2
    logits = ln(x, gf, bf) @ head
    return F.cross_entropy(logits.view(B * T, V), tgt.view(B * T))

  fn = torch.compile(fwd_loss) if compiled else fwd_loss

  def one_step():
    opt.zero_grad(set_to_none=True)
    lv = fn()
    lv.backward()
    opt.step()
    return lv

  torch.cuda.synchronize()
  t_c0 = time.perf_counter()
  loss0 = float(one_step().item())
  torch.cuda.synchronize()
  compile_s = time.perf_counter() - t_c0

  for _ in range(2):
    one_step()
  torch.cuda.synchronize()

  times = []
  lv = None
  for _ in range(steps):
    t0 = time.perf_counter()
    lv = one_step()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000.0)
  return {
    "compile_s": round(compile_s, 3),
    "median_ms": round(float(np.median(times)), 3),
    "loss0": round(loss0, 5),
    "loss_end": round(float(lv.item()), 5),
  }

# ---------------------------------------------------------------------------
# 3. JAX (jax.jit)
# ---------------------------------------------------------------------------
def bench_jax(cfg, params_np, tokens_np, targets_np, causal_np, steps=20):
  import jax
  import jax.numpy as jnp
  B, T, C, H, L, V = cfg
  D = C // H
  scale = jnp.float32(1.0 / np.sqrt(D))

  def ln(x, g, b):
    mu = jnp.mean(x, axis=2, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=2, keepdims=True)
    return (x - mu) / jnp.sqrt(var + jnp.float32(1e-5)) * g + b

  def loss_fn(params, tok, tgt, causal):
    wte, wpe, blocks, gf, bf, head = params
    x = wte[tok] + wpe
    for g1, b1, wq, wk, wv, wo, g2, b2, w1, c1, w2, c2 in blocks:
      nx = ln(x, g1, b1)
      q = (nx @ wq).reshape(B, T, H, D).transpose(0, 2, 1, 3)
      k = (nx @ wk).reshape(B, T, H, D).transpose(0, 2, 1, 3)
      v = (nx @ wv).reshape(B, T, H, D).transpose(0, 2, 1, 3)
      scores = (q @ k.transpose(0, 1, 3, 2)) * scale
      scores = jnp.where(causal, scores, jnp.float32(-1e9))
      m = jnp.max(scores, axis=-1, keepdims=True)
      e = jnp.exp(scores - m)
      w = e / jnp.sum(e, axis=-1, keepdims=True)
      attn = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, C) @ wo
      x = x + attn
      h = jnp.tanh(ln(x, g2, b2) @ w1 + c1)
      x = x + h @ w2 + c2
    logits = (ln(x, gf, bf) @ head).reshape(B * T, V)
    m = jnp.max(logits, axis=1, keepdims=True)
    z = logits - m
    lse = jnp.log(jnp.sum(jnp.exp(z), axis=1))
    picked = z[jnp.arange(B * T), tgt.reshape(B * T)]
    return -jnp.mean(picked - lse)

  @jax.jit
  def train_step(params, m, v, tok, tgt, causal, step):
    lv, grads = jax.value_and_grad(loss_fn)(params, tok, tgt, causal)
    lr = jnp.float32(3e-3); b1 = jnp.float32(0.9); b2 = jnp.float32(0.99)
    bc1 = jnp.float32(1.0) - b1 ** step
    bc2 = jnp.float32(1.0) - b2 ** step
    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (jnp.float32(1.0) - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (jnp.float32(1.0) - b2) * g * g, v, grads)
    params = jax.tree_util.tree_map(lambda p, mi, vi: p - (lr * mi / bc1) / (jnp.sqrt(vi / bc2) + jnp.float32(1e-8)), params, m, v)
    return lv, params, m, v

  params = jax.device_put(tree_copy(params_np))
  m = jax.device_put(tree_zeros(params_np))
  v = jax.device_put(tree_zeros(params_np))
  tok = jax.device_put(tokens_np)
  tgt = jax.device_put(targets_np)
  causal = jax.device_put(causal_np)

  t_c0 = time.perf_counter()
  loss0, params, m, v = train_step(params, m, v, tok, tgt, causal, jnp.int32(1))
  jax.block_until_ready(loss0)
  compile_s = time.perf_counter() - t_c0

  for s in (2, 3):
    lv, params, m, v = train_step(params, m, v, tok, tgt, causal, jnp.int32(s))
  jax.block_until_ready(lv)

  times = []
  for s in range(4, 4 + steps):
    t0 = time.perf_counter()
    lv, params, m, v = train_step(params, m, v, tok, tgt, causal, jnp.int32(s))
    jax.block_until_ready(lv)
    times.append((time.perf_counter() - t0) * 1000.0)
  return {
    "compile_s": round(compile_s, 3),
    "median_ms": round(float(np.median(times)), 3),
    "loss0": round(float(loss0), 5),
    "loss_end": round(float(lv), 5),
  }

def main():
  prof = parakeet.target_profile("cuda")
  print("Parakeet CUDA TargetProfile:", prof, flush=True)
  results = {"profile": str(prof), "configs": {}}
  configs = [
    ("small_B16_T64_C128_L2", (16, 64, 128, 4, 2, 65)),
    ("med_B32_T128_C256_L4", (32, 128, 256, 4, 4, 65)),
  ]
  for name, cfg in configs:
    B, T, C, H, L, V = cfg
    rng = np.random.default_rng(42)
    params = init_gpt(V, T, C, L, seed=0)
    tokens = rng.integers(0, V, size=(B, T), dtype=np.int64)
    targets = rng.integers(0, V, size=(B, T), dtype=np.int64)
    causal = np.tril(np.ones((T, T), dtype=bool))
    print(f"\n=== {name} ===", flush=True)
    row = {}
    for label, fn in [
      ("parakeet_fp32", lambda: bench_parakeet(cfg, params, tokens, targets, causal, tensor_cores=False)),
      ("parakeet_tf32_wmma", lambda: bench_parakeet(cfg, params, tokens, targets, causal, tensor_cores=True)),
      ("torch_eager", lambda: bench_torch(cfg, params, tokens, targets, causal, compiled=False)),
      ("torch_compile", lambda: bench_torch(cfg, params, tokens, targets, causal, compiled=True)),
      ("jax_jit", lambda: bench_jax(cfg, params, tokens, targets, causal)),
    ]:
      try:
        r = fn()
        row[label] = r
        print(f"  {label:20s}: {r}", flush=True)
      except Exception as e:
        import traceback
        row[label] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  {label:20s}: ERROR {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    results["configs"][name] = row
  print("\nSUMMARY_JSON:" + json.dumps(results), flush=True)

if __name__ == "__main__":
  main()
"""


@app.function(image=image, gpu=GPU, timeout=30 * 60)
def run_bench():
  import pathlib
  import subprocess
  import sys

  repo = pathlib.Path(__file__).resolve().parents[1]
  py = sys.executable
  subprocess.run(["bash", "-c", "nvidia-smi; nproc; python --version"], check=False)
  subprocess.run([py, "-m", "pip", "install", "-q", "-e", "."], cwd=repo, check=True)
  script = repo / "_nanogpt_bench.py"
  script.write_text(BENCH_SCRIPT)
  env = dict(os.environ, PYTHONPATH=str(repo))
  subprocess.run([py, str(script)], cwd=repo, env=env, check=True)


@app.local_entrypoint()
def main():
  run_bench.remote()
