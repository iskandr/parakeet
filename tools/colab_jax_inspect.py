"""Inspect JAX/XLA optimized HLO, fused kernels, cuBLAS/Triton calls, and PTX on NVIDIA A100.

Usage::

    runplz --outputs-dir /tmp/pk-jax-inspect colab tools/colab_jax_inspect.py
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from runplz import App, Image

GPU = os.environ.get("PK_COLAB_GPU", "A100")
app = App("parakeet-jax-inspect")
image = Image.from_registry("python:3.12")

INSPECT_SCRIPT = r"""
import collections
import json
import os
import pathlib
import re
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "tensorfloat32")
print("JAX:", jax.__version__, "devices:", jax.devices())

out_dir = pathlib.Path(os.environ.get("RUNPLZ_OUT", "/content/out"))
out_dir.mkdir(parents=True, exist_ok=True)
summary = {}

def analyze_hlo(name: str, fn, args, dump_prefix: str):
  compiled = jax.jit(fn).lower(*args).compile()
  hlo = compiled.as_text()
  (out_dir / f"hlo_{dump_prefix}.txt").write_text(hlo)

  # Parse entry computation instructions (custom-calls, fusions, command_buffers)
  custom_calls = collections.Counter()
  fusions = collections.Counter()
  triton_fusions = 0
  command_buffers = 0
  for line in hlo.splitlines():
    m_cc = re.search(r'custom_call_target="([^"]+)"', line)
    if m_cc:
      custom_calls[m_cc.group(1)] += 1
    if "kind=k" in line and "fusion(" in line:
      mk = re.search(r'kind=(k\w+)', line)
      if mk:
        fusions[mk.group(1)] += 1
      if "__triton" in line or "triton_gemm" in line:
        triton_fusions += 1
    if "command_buffer" in line and "ENTRY" not in line and "call(" in line:
      command_buffers += 1

  # Warm and time
  for _ in range(3):
    jax.block_until_ready(compiled(*args))
  ts = []
  for _ in range(20):
    t0 = time.perf_counter()
    jax.block_until_ready(compiled(*args))
    ts.append((time.perf_counter() - t0) * 1e3)
  ts.sort()
  med = ts[len(ts) // 2]

  # Also extract memory / cost analysis if available
  mem = compiled.memory_analysis()
  mem_info = {
    "temp_mb": round(getattr(mem, "temp_size_in_bytes", 0) / 1e6, 2),
    "arg_mb": round(getattr(mem, "argument_size_in_bytes", 0) / 1e6, 2),
    "out_mb": round(getattr(mem, "output_size_in_bytes", 0) / 1e6, 2),
  } if mem is not None else {}

  info = {
    "median_ms": round(med, 3),
    "hlo_lines": len(hlo.splitlines()),
    "custom_calls": dict(custom_calls),
    "fusions": dict(fusions),
    "triton_fusions": triton_fusions,
    "command_buffers": command_buffers,
    "memory": mem_info,
  }
  summary[name] = info
  print(f"\n=== {name} ({med:.3f} ms) ===")
  print("  custom_calls:", dict(custom_calls))
  print("  fusions:", dict(fusions), "triton:", triton_fusions, "cmd_buffers:", command_buffers)
  print("  memory:", mem_info)
  # Print ENTRY computation snippet
  in_entry = False
  entry_lines = []
  for line in hlo.splitlines():
    if line.startswith("ENTRY "):
      in_entry = True
    if in_entry:
      entry_lines.append(line)
      if line.strip() == "}":
        break
  print("  ENTRY lines:", len(entry_lines))
  for el in entry_lines[:40]:
    print("   ", el)

rng = np.random.default_rng(123)

# 1. Large CCE (4096 x 512 x 65536)
h = jnp.asarray(rng.standard_normal((4096, 512), dtype=np.float32) * 0.05)
w = jnp.asarray(rng.standard_normal((512, 65536), dtype=np.float32) * 0.05)
tgt = jnp.asarray(rng.integers(65536, size=4096, dtype=np.int64))
def jax_cce(x, y, t):
  z = x @ y
  z = z - z.max(1, keepdims=True)
  return (jnp.log(jnp.exp(z).sum(1)) - z[jnp.arange(x.shape[0]), t]).mean()
analyze_hlo("cce_large", jax_cce, (h, w, tgt), "cce_large")

# 2. Large Causal Attention (4 x 16 x 4096 x 64)
q = jnp.asarray(rng.standard_normal((4, 16, 4096, 64), dtype=np.float32) * 0.2)
k = jnp.asarray(rng.standard_normal((4, 16, 4096, 64), dtype=np.float32) * 0.2)
v = jnp.asarray(rng.standard_normal((4, 16, 4096, 64), dtype=np.float32) * 0.2)
def jax_causal(q, k, v):
  scores = jnp.where(jnp.tri(q.shape[-2], dtype=bool), (q @ k.swapaxes(-1, -2)) / 8.0, -float("inf"))
  probs = jnp.exp(scores - scores.max(-1, keepdims=True))
  return (probs / probs.sum(-1, keepdims=True)) @ v
analyze_hlo("causal_large", jax_causal, (q, k, v), "causal_large")

# 3. Medium nanoGPT training step (B=32, T=128, C=256, H=4, L=4)
from benchmarks.nanogpt_bench import DEFAULT_CONFIGS, init_gpt, _model_to_pytree
cfg = DEFAULT_CONFIGS["medium"]
ini = init_gpt(cfg, seed=0)
head_dim = cfg.width // cfg.heads
scale = jnp.float32(head_dim ** -0.5)

def _layer_norm(x, params):
  gain, bias = params
  mu = jnp.mean(x, axis=2, keepdims=True)
  var = jnp.mean((x - mu) ** 2, axis=2, keepdims=True)
  return (x - mu) / jnp.sqrt(var + jnp.float32(1e-5)) * gain + bias

def _gpt_loss(params, tok, tgt, mask):
  x = params["wte"][tok] + params["wpe"]
  for b in params["blocks"]:
    nx = _layer_norm(x, b["ln1"])
    q = (nx @ b["wq"]).reshape(cfg.batch, cfg.context, cfg.heads, head_dim).transpose(0, 2, 1, 3)
    k = (nx @ b["wk"]).reshape(cfg.batch, cfg.context, cfg.heads, head_dim).transpose(0, 2, 1, 3)
    v = (nx @ b["wv"]).reshape(cfg.batch, cfg.context, cfg.heads, head_dim).transpose(0, 2, 1, 3)
    scores = jnp.where(mask, (q @ k.transpose(0, 1, 3, 2)) * scale, jnp.float32(-1e9))
    weights = jax.nn.softmax(scores, axis=-1)
    attn = (weights @ v).transpose(0, 2, 1, 3).reshape(cfg.batch, cfg.context, cfg.width) @ b["wo"]
    x = x + attn
    h = jnp.tanh(_layer_norm(x, b["ln2"]) @ b["w1"] + b["b1"])
    x = x + h @ b["w2"] + b["b2"]
  logits = (_layer_norm(x, params["ln_f"]) @ params["head"]).reshape(-1, cfg.vocab)
  shifted = logits - jnp.max(logits, axis=1, keepdims=True)
  lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=1))
  picked = shifted[jnp.arange(logits.shape[0]), tgt.reshape(-1)]
  return -jnp.mean(picked - lse)

def step_fn(params, m, v, tok, tgt, mask, step):
  lv, grads = jax.value_and_grad(_gpt_loss)(params, tok, tgt, mask)
  lr = jnp.float32(3e-3)
  b1 = jnp.float32(0.9)
  b2 = jnp.float32(0.99)
  bc1 = jnp.float32(1.0) - b1 ** step
  bc2 = jnp.float32(1.0) - b2 ** step
  m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (jnp.float32(1.0) - b1) * g, m, grads)
  v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (jnp.float32(1.0) - b2) * g * g, v, grads)
  params = jax.tree_util.tree_map(
    lambda p, mi, vi: p - (lr * mi / bc1) / (jnp.sqrt(vi / bc2) + jnp.float32(1e-8)),
    params, m, v,
  )
  return lv, params, m, v

pytree = _model_to_pytree(ini)
j_params = jax.device_put(pytree)
j_m = jax.device_put(jax.tree_util.tree_map(np.zeros_like, pytree))
j_v = jax.device_put(jax.tree_util.tree_map(np.zeros_like, pytree))
tok = jnp.asarray(rng.integers(0, cfg.vocab, size=(cfg.batch, cfg.context), dtype=np.int64))
tgt_g = jnp.asarray(rng.integers(0, cfg.vocab, size=(cfg.batch, cfg.context), dtype=np.int64))
cas = jnp.asarray(np.tril(np.ones((cfg.context, cfg.context), dtype=bool)))
analyze_hlo("nanogpt_med", step_fn, (j_params, j_m, j_v, tok, tgt_g, cas, jnp.int32(1)), "nanogpt_med")

(out_dir / "jax_hlo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
"""


@app.function(image=image, gpu=GPU, timeout=15 * 60)
def inspect():
  repo = pathlib.Path(__file__).resolve().parents[1]
  py = sys.executable
  env = dict(os.environ, PYTHONPATH=str(repo))
  subprocess.run([py, "-m", "pip", "install", "-q", "-e", "."], cwd=repo, check=True)
  script = repo / "_jax_inspect_tmp.py"
  script.write_text(INSPECT_SCRIPT)
  try:
    subprocess.run([py, str(script)], cwd=repo, env=env, check=True)
  finally:
    script.unlink(missing_ok=True)


@app.local_entrypoint()
def main():
  inspect.remote()
