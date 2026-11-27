"""Cross-framework nanoGPT training benchmark.

Compares multiple Parakeet implementations against PyTorch and JAX on identical
float32 weights and batches, checking exact loss agreement before timing:

Parakeet implementations (showing the main design choices):
  - ``parakeet_batched_fp32`` / ``parakeet_batched_tf32``:
      4D batched ``(B, H, T, D)`` attention with Adam compiled inside ``@par.jit``
      so each step executes with zero host round-trips.
  - ``parakeet_mapped_fp32`` / ``parakeet_mapped_tf32``:
      Per-sequence ``par.map`` over single-sequence attention (same model style
      as ``examples/nanogpt.py``) with compiled Adam.
  - ``parakeet_host_adam_fp32``:
      Compiled ``value_and_grad`` paired with a plain Python Adam loop, showing
      the effect of leaving the optimizer outside the compiled step.

Baselines:
  - ``torch_eager_fp32``, ``torch_compile_fp32``, ``torch_compile_tf32``:
      Explicit matmul + masked softmax attention in an ``nn.Module``.
  - ``torch_sdpa_compile_tf32``:
      Same ``nn.Module`` using ``F.scaled_dot_product_attention(..., is_causal=True)``.
  - ``jax_jit_fp32``, ``jax_jit_tf32``:
      Compiled ``jax.value_and_grad`` + pytree Adam update.

Run directly::

    python -m benchmarks.nanogpt_bench --device cuda --steps 20
    python -m benchmarks.nanogpt_bench --device cpu --configs small --steps 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np

import parakeet as par


@dataclass(frozen=True)
class GPTConfig:
  name: str
  batch: int
  context: int
  width: int
  heads: int
  layers: int
  vocab: int = 65


DEFAULT_CONFIGS: dict[str, GPTConfig] = {
  "small": GPTConfig("small_B16_T64_C128_L2", batch=16, context=64, width=128, heads=4, layers=2),
  "medium": GPTConfig("med_B32_T128_C256_L4", batch=32, context=128, width=256, heads=4, layers=4),
}


def tree_map(fn, *trees):
  first = trees[0]
  if isinstance(first, tuple):
    return tuple(tree_map(fn, *parts) for parts in zip(*trees))
  if is_dataclass(first):
    return type(first)(*(tree_map(fn, *(getattr(t, fl.name) for t in trees)) for fl in fields(first)))
  return fn(*trees)


def _adam_leaf(param, grad, m, v, lr, b1, b2, bc1, bc2):
  one = np.float32(1.0)
  m_next = b1 * m + (one - b1) * grad
  v_next = b2 * v + (one - b2) * grad * grad
  step = (lr * m_next / bc1) / (np.sqrt(v_next / bc2) + np.float32(1e-8))
  return param - step, m_next, v_next


@dataclass
class LayerNorm:
  gain: np.ndarray
  bias: np.ndarray

  def __call__(self, x, axis=-1):
    mu = np.mean(x, axis=axis, keepdims=True)
    var = np.mean((x - mu) ** np.float32(2.0), axis=axis, keepdims=True)
    return (x - mu) / np.sqrt(var + np.float32(1e-5)) * self.gain + self.bias

  def adam_update(self, grad, m, v, lr, b1, b2, bc1, bc2):
    gain, mg, vg = _adam_leaf(self.gain, grad.gain, m.gain, v.gain, lr, b1, b2, bc1, bc2)
    bias, mb, vb = _adam_leaf(self.bias, grad.bias, m.bias, v.bias, lr, b1, b2, bc1, bc2)
    return LayerNorm(gain, bias), LayerNorm(mg, mb), LayerNorm(vg, vb)


@dataclass
class Block:
  ln1: LayerNorm
  wq: np.ndarray
  wk: np.ndarray
  wv: np.ndarray
  wo: np.ndarray
  ln2: LayerNorm
  w1: np.ndarray
  b1: np.ndarray
  w2: np.ndarray
  b2: np.ndarray

  def attention_batched(self, x, causal, n_head):
    batch, seq, width = x.shape
    head_dim = width // n_head
    q = (x @ self.wq).reshape((batch, seq, n_head, head_dim)).transpose((0, 2, 1, 3))
    k = (x @ self.wk).reshape((batch, seq, n_head, head_dim)).transpose((0, 2, 1, 3))
    v = (x @ self.wv).reshape((batch, seq, n_head, head_dim)).transpose((0, 2, 1, 3))
    scores = (q @ k.transpose((0, 1, 3, 2))) * np.float32(1.0 / np.sqrt(head_dim))
    masked = np.where(causal, scores, np.float32(-1e9))
    shifted = np.exp(masked - np.max(masked, axis=3, keepdims=True))
    weights = shifted / np.sum(shifted, axis=3, keepdims=True)
    out = (weights @ v).transpose((0, 2, 1, 3)).reshape((batch, seq, width))
    return out @ self.wo

  def attention_sequence(self, x, causal, n_head):
    seq, width = x.shape
    head_dim = width // n_head
    q = (x @ self.wq).reshape((seq, n_head, head_dim)).transpose((1, 0, 2))
    k = (x @ self.wk).reshape((seq, n_head, head_dim)).transpose((1, 0, 2))
    v = (x @ self.wv).reshape((seq, n_head, head_dim)).transpose((1, 0, 2))
    scores = (q @ k.transpose((0, 2, 1))) * np.float32(1.0 / np.sqrt(head_dim))
    masked = np.where(causal, scores, np.float32(-1e9))
    shifted = np.exp(masked - np.max(masked, axis=2, keepdims=True))
    weights = shifted / np.sum(shifted, axis=2, keepdims=True)
    out = (weights @ v).transpose((1, 0, 2)).reshape((seq, width))
    return out @ self.wo

  def forward_batched(self, x, causal, n_head):
    x = x + self.attention_batched(self.ln1(x, axis=2), causal, n_head)
    h = np.tanh(self.ln2(x, axis=2) @ self.w1 + self.b1)
    return x + h @ self.w2 + self.b2

  def forward_sequence(self, x, causal, n_head):
    x = x + self.attention_sequence(self.ln1(x, axis=1), causal, n_head)
    h = np.tanh(self.ln2(x, axis=1) @ self.w1 + self.b1)
    return x + h @ self.w2 + self.b2

  def adam_update(self, grad, m, v, lr, b1, b2, bc1, bc2):
    ln1, m_ln1, v_ln1 = self.ln1.adam_update(grad.ln1, m.ln1, v.ln1, lr, b1, b2, bc1, bc2)
    wq, m_wq, v_wq = _adam_leaf(self.wq, grad.wq, m.wq, v.wq, lr, b1, b2, bc1, bc2)
    wk, m_wk, v_wk = _adam_leaf(self.wk, grad.wk, m.wk, v.wk, lr, b1, b2, bc1, bc2)
    wv, m_wv, v_wv = _adam_leaf(self.wv, grad.wv, m.wv, v.wv, lr, b1, b2, bc1, bc2)
    wo, m_wo, v_wo = _adam_leaf(self.wo, grad.wo, m.wo, v.wo, lr, b1, b2, bc1, bc2)
    ln2, m_ln2, v_ln2 = self.ln2.adam_update(grad.ln2, m.ln2, v.ln2, lr, b1, b2, bc1, bc2)
    w1, m_w1, v_w1 = _adam_leaf(self.w1, grad.w1, m.w1, v.w1, lr, b1, b2, bc1, bc2)
    b1_p, m_b1, v_b1 = _adam_leaf(self.b1, grad.b1, m.b1, v.b1, lr, b1, b2, bc1, bc2)
    w2, m_w2, v_w2 = _adam_leaf(self.w2, grad.w2, m.w2, v.w2, lr, b1, b2, bc1, bc2)
    b2_p, m_b2, v_b2 = _adam_leaf(self.b2, grad.b2, m.b2, v.b2, lr, b1, b2, bc1, bc2)
    return (
      Block(ln1, wq, wk, wv, wo, ln2, w1, b1_p, w2, b2_p),
      Block(m_ln1, m_wq, m_wk, m_wv, m_wo, m_ln2, m_w1, m_b1, m_w2, m_b2),
      Block(v_ln1, v_wq, v_wk, v_wv, v_wo, v_ln2, v_w1, v_b1, v_w2, v_b2),
    )


@dataclass
class NanoGPT:
  wte: np.ndarray
  wpe: np.ndarray
  blocks: tuple[Block, ...]
  ln_f: LayerNorm
  head: np.ndarray

  def adam_update(self, grad, m, v, lr, b1, b2, bc1, bc2):
    wte, m_wte, v_wte = _adam_leaf(self.wte, grad.wte, m.wte, v.wte, lr, b1, b2, bc1, bc2)
    wpe, m_wpe, v_wpe = _adam_leaf(self.wpe, grad.wpe, m.wpe, v.wpe, lr, b1, b2, bc1, bc2)
    ln_f, m_ln_f, v_ln_f = self.ln_f.adam_update(grad.ln_f, m.ln_f, v.ln_f, lr, b1, b2, bc1, bc2)
    head, m_head, v_head = _adam_leaf(self.head, grad.head, m.head, v.head, lr, b1, b2, bc1, bc2)

    blocks_next = ()
    m_blocks_next = ()
    v_blocks_next = ()
    for blk, g_blk, m_blk, v_blk in zip(self.blocks, grad.blocks, m.blocks, v.blocks):
      b_new, mb_new, vb_new = blk.adam_update(g_blk, m_blk, v_blk, lr, b1, b2, bc1, bc2)
      blocks_next = blocks_next + (b_new,)
      m_blocks_next = m_blocks_next + (mb_new,)
      v_blocks_next = v_blocks_next + (vb_new,)

    return (
      NanoGPT(wte, wpe, blocks_next, ln_f, head),
      NanoGPT(m_wte, m_wpe, m_blocks_next, m_ln_f, m_head),
      NanoGPT(v_wte, v_wpe, v_blocks_next, v_ln_f, v_head),
    )


def init_gpt(cfg: GPTConfig, seed: int = 0) -> NanoGPT:
  rng = np.random.default_rng(seed)

  def weight(*shape):
    return rng.normal(0.0, 0.02, shape).astype(np.float32)

  def norm():
    return LayerNorm(np.ones(cfg.width, dtype=np.float32), np.zeros(cfg.width, dtype=np.float32))

  def make_block():
    return Block(
      ln1=norm(),
      wq=weight(cfg.width, cfg.width),
      wk=weight(cfg.width, cfg.width),
      wv=weight(cfg.width, cfg.width),
      wo=weight(cfg.width, cfg.width),
      ln2=norm(),
      w1=weight(cfg.width, 4 * cfg.width),
      b1=np.zeros(4 * cfg.width, dtype=np.float32),
      w2=weight(4 * cfg.width, cfg.width),
      b2=np.zeros(cfg.width, dtype=np.float32),
    )

  return NanoGPT(
    wte=weight(cfg.vocab, cfg.width),
    wpe=weight(cfg.context, cfg.width),
    blocks=tuple(make_block() for _ in range(cfg.layers)),
    ln_f=norm(),
    head=weight(cfg.width, cfg.vocab),
  )


# ---------------------------------------------------------------------------
# Parakeet implementations
# ---------------------------------------------------------------------------


def _cross_entropy(logits, targets):
  shifted = logits - np.max(logits, axis=1, keepdims=True)
  log_denom = np.log(np.sum(np.exp(shifted), axis=1))
  picked = shifted[np.arange(logits.shape[0]), targets]
  return -np.mean(picked - log_denom)


def loss_batched(model: NanoGPT, tokens, targets, causal, n_head):
  batch, seq = tokens.shape
  width = model.wte.shape[1]
  x = model.wte[tokens.reshape((batch * seq,))].reshape((batch, seq, width)) + model.wpe
  for blk in model.blocks:
    x = blk.forward_batched(x, causal, n_head)
  logits = model.ln_f(x, axis=2) @ model.head
  return _cross_entropy(logits.reshape((batch * seq, logits.shape[2])), targets.reshape((batch * seq,)))


def loss_mapped(model: NanoGPT, tokens, targets, causal, n_head):
  def sequence_loss(tok_row, tgt_row):
    x = model.wte[tok_row] + model.wpe
    for blk in model.blocks:
      x = blk.forward_sequence(x, causal, n_head)
    logits = model.ln_f(x, axis=1) @ model.head
    return _cross_entropy(logits, tgt_row)

  return np.mean(par.map(sequence_loss, tokens, targets))


_vg_batched = par.value_and_grad(loss_batched)
_vg_mapped = par.value_and_grad(loss_mapped)


@par.jit
def step_batched_compiled(model, m, v, tokens, targets, causal, n_head, lr, b1, b2, bc1, bc2):
  loss_val, grads = _vg_batched(model, tokens, targets, causal, n_head)
  model, m, v = model.adam_update(grads, m, v, lr, b1, b2, bc1, bc2)
  return loss_val, model, m, v


@par.jit
def step_mapped_compiled(model, m, v, tokens, targets, causal, n_head, lr, b1, b2, bc1, bc2):
  loss_val, grads = _vg_mapped(model, tokens, targets, causal, n_head)
  model, m, v = model.adam_update(grads, m, v, lr, b1, b2, bc1, bc2)
  return loss_val, model, m, v


def _adam_bias_correction(step: int, b1: float = 0.9, b2: float = 0.99):
  return np.float32(1.0 - b1**step), np.float32(1.0 - b2**step)


def _summarize_times(times_ms: list[float], compile_s: float, loss0: float, loss_end: float) -> dict:
  arr = np.asarray(times_ms, dtype=np.float64)
  q25, q75 = np.percentile(arr, [25, 75])
  return {
    "compile_s": round(compile_s, 3),
    "median_ms": round(float(statistics.median(times_ms)), 3),
    "iqr_ms": round(float(q75 - q25), 3),
    "samples_ms": [round(t, 3) for t in times_ms],
    "loss0": round(float(loss0), 5),
    "loss_end": round(float(loss_end), 5),
  }


def bench_parakeet(
  cfg: GPTConfig,
  initial: NanoGPT,
  tokens: np.ndarray,
  targets: np.ndarray,
  causal: np.ndarray,
  *,
  variant: str = "batched",
  backend: str = "cuda",
  tensor_cores: bool = False,
  steps: int = 20,
) -> dict:
  def sync():
    return None

  if backend == "cuda":
    import cupy

    sync = cupy.cuda.Stream.null.synchronize

  lr = np.float32(3e-3)
  b1 = np.float32(0.9)
  b2 = np.float32(0.99)

  opts = dict(backend=backend, device_threshold=1, device_fallback="silent", verify=False, tensor_cores=tensor_cores)
  with par.options(**opts):
    model = tree_map(np.copy, initial)
    m = tree_map(np.zeros_like, initial)
    v = tree_map(np.zeros_like, initial)
    tok, tgt, cas = tokens, targets, causal

    if backend == "cuda" and variant != "host_adam":
      model = par.to_device(model, backend="cuda")
      m = par.to_device(m, backend="cuda")
      v = par.to_device(v, backend="cuda")
      tok = par.to_device(tokens, backend="cuda")
      tgt = par.to_device(targets, backend="cuda")
      cas = par.to_device(causal, backend="cuda")

    if variant == "batched":

      def run_step(step_idx, cur_model, cur_m, cur_v):
        bc1, bc2 = _adam_bias_correction(step_idx)
        return step_batched_compiled(cur_model, cur_m, cur_v, tok, tgt, cas, cfg.heads, lr, b1, b2, bc1, bc2)
    elif variant == "mapped":

      def run_step(step_idx, cur_model, cur_m, cur_v):
        bc1, bc2 = _adam_bias_correction(step_idx)
        return step_mapped_compiled(cur_model, cur_m, cur_v, tok, tgt, cas, cfg.heads, lr, b1, b2, bc1, bc2)
    elif variant == "host_adam":

      def run_step(step_idx, cur_model, cur_m, cur_v):
        bc1, bc2 = _adam_bias_correction(step_idx)
        lv, grads = _vg_batched(cur_model, tok, tgt, cas, cfg.heads)
        cur_model, cur_m, cur_v = cur_model.adam_update(grads, cur_m, cur_v, lr, b1, b2, bc1, bc2)
        return lv, cur_model, cur_m, cur_v
    else:
      raise ValueError(f"unknown Parakeet variant: {variant}")

    t_compile = time.perf_counter()
    loss0, model, m, v = run_step(1, model, m, v)
    sync()
    compile_s = time.perf_counter() - t_compile

    for s in (2, 3):
      loss_end, model, m, v = run_step(s, model, m, v)
    sync()

    dev = par.api.get_backend("cuda").device() if backend == "cuda" else None
    before = (dev.uploads, dev.downloads, dev.launches) if dev is not None else None

    times_ms = []
    for s in range(4, 4 + steps):
      t0 = time.perf_counter()
      loss_end, model, m, v = run_step(s, model, m, v)
      sync()
      times_ms.append((time.perf_counter() - t0) * 1e3)

    out = _summarize_times(times_ms, compile_s, float(loss0), float(loss_end))
    if dev is not None and before is not None:
      u0, d0, l0 = before
      out["uploads_per_step"] = round((dev.uploads - u0) / steps, 2)
      out["downloads_per_step"] = round((dev.downloads - d0) / steps, 2)
      out["launches_per_step"] = round((dev.launches - l0) / steps, 1)
    return out


# ---------------------------------------------------------------------------
# PyTorch baselines
# ---------------------------------------------------------------------------


def bench_torch(
  cfg: GPTConfig,
  initial: NanoGPT,
  tokens: np.ndarray,
  targets: np.ndarray,
  causal: np.ndarray,
  *,
  device_name: str = "cuda",
  compiled: bool = False,
  tf32: bool = False,
  use_sdpa: bool = False,
  steps: int = 20,
) -> dict:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  if device_name == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

  device = torch.device(device_name)
  sync = torch.cuda.synchronize if device_name == "cuda" else (lambda: None)

  def as_param(arr):
    return nn.Parameter(torch.tensor(arr, device=device, dtype=torch.float32))

  class TorchLayerNorm(nn.Module):
    def __init__(self, src: LayerNorm):
      super().__init__()
      self.gain = as_param(src.gain)
      self.bias = as_param(src.bias)

    def forward(self, x):
      mu = x.mean(dim=-1, keepdim=True)
      var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
      return (x - mu) / torch.sqrt(var + 1e-5) * self.gain + self.bias

  class TorchBlock(nn.Module):
    def __init__(self, src: Block):
      super().__init__()
      self.ln1 = TorchLayerNorm(src.ln1)
      self.wq = as_param(src.wq)
      self.wk = as_param(src.wk)
      self.wv = as_param(src.wv)
      self.wo = as_param(src.wo)
      self.ln2 = TorchLayerNorm(src.ln2)
      self.w1 = as_param(src.w1)
      self.b1 = as_param(src.b1)
      self.w2 = as_param(src.w2)
      self.b2 = as_param(src.b2)

    def forward(self, x, mask):
      batch, seq, width = x.shape
      head_dim = width // cfg.heads
      nx = self.ln1(x)
      q = (nx @ self.wq).view(batch, seq, cfg.heads, head_dim).transpose(1, 2)
      k = (nx @ self.wk).view(batch, seq, cfg.heads, head_dim).transpose(1, 2)
      v = (nx @ self.wv).view(batch, seq, cfg.heads, head_dim).transpose(1, 2)
      if use_sdpa:
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
      else:
        scores = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
        scores = torch.where(mask, scores, torch.tensor(-1e9, device=device, dtype=torch.float32))
        ctx = torch.softmax(scores, dim=-1) @ v
      x = x + ctx.transpose(1, 2).contiguous().view(batch, seq, width) @ self.wo
      h = torch.tanh(self.ln2(x) @ self.w1 + self.b1)
      return x + h @ self.w2 + self.b2

  class TorchNanoGPT(nn.Module):
    def __init__(self, src: NanoGPT):
      super().__init__()
      self.wte = as_param(src.wte)
      self.wpe = as_param(src.wpe)
      self.blocks = nn.ModuleList([TorchBlock(b) for b in src.blocks])
      self.ln_f = TorchLayerNorm(src.ln_f)
      self.head = as_param(src.head)

    def forward(self, tok, tgt, mask):
      x = self.wte[tok] + self.wpe
      for blk in self.blocks:
        x = blk(x, mask)
      logits = self.ln_f(x) @ self.head
      return F.cross_entropy(logits.view(-1, cfg.vocab), tgt.view(-1))

  model = TorchNanoGPT(initial)
  tok = torch.tensor(tokens, device=device, dtype=torch.long)
  tgt = torch.tensor(targets, device=device, dtype=torch.long)
  mask = torch.tensor(causal, device=device, dtype=torch.bool)
  opt = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.99), eps=1e-8)

  forward_fn = torch.compile(model) if compiled else model

  def one_step():
    opt.zero_grad(set_to_none=True)
    lv = forward_fn(tok, tgt, mask)
    lv.backward()
    opt.step()
    return lv

  sync()
  t_compile = time.perf_counter()
  loss0 = float(one_step().item())
  sync()
  compile_s = time.perf_counter() - t_compile

  for _ in range(2):
    loss_end = one_step()
  sync()

  times_ms = []
  for _ in range(steps):
    t0 = time.perf_counter()
    loss_end = one_step()
    sync()
    times_ms.append((time.perf_counter() - t0) * 1e3)

  return _summarize_times(times_ms, compile_s, loss0, float(loss_end.item()))


# ---------------------------------------------------------------------------
# JAX baselines
# ---------------------------------------------------------------------------


def _model_to_pytree(model: NanoGPT) -> dict:
  return {
    "wte": model.wte,
    "wpe": model.wpe,
    "ln_f": (model.ln_f.gain, model.ln_f.bias),
    "head": model.head,
    "blocks": tuple(
      {
        "ln1": (b.ln1.gain, b.ln1.bias),
        "wq": b.wq,
        "wk": b.wk,
        "wv": b.wv,
        "wo": b.wo,
        "ln2": (b.ln2.gain, b.ln2.bias),
        "w1": b.w1,
        "b1": b.b1,
        "w2": b.w2,
        "b2": b.b2,
      }
      for b in model.blocks
    ),
  }


def bench_jax(
  cfg: GPTConfig,
  initial: NanoGPT,
  tokens: np.ndarray,
  targets: np.ndarray,
  causal: np.ndarray,
  *,
  tf32: bool = False,
  steps: int = 20,
) -> dict:
  import jax
  import jax.numpy as jnp

  jax.config.update("jax_default_matmul_precision", "tensorfloat32" if tf32 else "highest")
  head_dim = cfg.width // cfg.heads
  scale = jnp.float32(head_dim**-0.5)

  def layer_norm(x, params):
    gain, bias = params
    mu = jnp.mean(x, axis=2, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=2, keepdims=True)
    return (x - mu) / jnp.sqrt(var + jnp.float32(1e-5)) * gain + bias

  def loss_fn(params, tok, tgt, mask):
    x = params["wte"][tok] + params["wpe"]
    for b in params["blocks"]:
      nx = layer_norm(x, b["ln1"])
      q = (nx @ b["wq"]).reshape(cfg.batch, cfg.context, cfg.heads, head_dim).transpose(0, 2, 1, 3)
      k = (nx @ b["wk"]).reshape(cfg.batch, cfg.context, cfg.heads, head_dim).transpose(0, 2, 1, 3)
      v = (nx @ b["wv"]).reshape(cfg.batch, cfg.context, cfg.heads, head_dim).transpose(0, 2, 1, 3)
      scores = jnp.where(mask, (q @ k.transpose(0, 1, 3, 2)) * scale, jnp.float32(-1e9))
      weights = jax.nn.softmax(scores, axis=-1)
      attn = (weights @ v).transpose(0, 2, 1, 3).reshape(cfg.batch, cfg.context, cfg.width) @ b["wo"]
      x = x + attn
      h = jnp.tanh(layer_norm(x, b["ln2"]) @ b["w1"] + b["b1"])
      x = x + h @ b["w2"] + b["b2"]

    logits = (layer_norm(x, params["ln_f"]) @ params["head"]).reshape(-1, cfg.vocab)
    shifted = logits - jnp.max(logits, axis=1, keepdims=True)
    lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=1))
    picked = shifted[jnp.arange(logits.shape[0]), tgt.reshape(-1)]
    return -jnp.mean(picked - lse)

  @jax.jit
  def train_step(params, m, v, tok, tgt, mask, step):
    lv, grads = jax.value_and_grad(loss_fn)(params, tok, tgt, mask)
    lr = jnp.float32(3e-3)
    b1 = jnp.float32(0.9)
    b2 = jnp.float32(0.99)
    bc1 = jnp.float32(1.0) - b1**step
    bc2 = jnp.float32(1.0) - b2**step
    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (jnp.float32(1.0) - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (jnp.float32(1.0) - b2) * g * g, v, grads)
    params = jax.tree_util.tree_map(
      lambda p, mi, vi: p - (lr * mi / bc1) / (jnp.sqrt(vi / bc2) + jnp.float32(1e-8)),
      params,
      m,
      v,
    )
    return lv, params, m, v

  pytree = _model_to_pytree(initial)
  params = jax.device_put(pytree)
  m = jax.device_put(jax.tree_util.tree_map(np.zeros_like, pytree))
  v = jax.device_put(jax.tree_util.tree_map(np.zeros_like, pytree))
  tok = jax.device_put(tokens)
  tgt = jax.device_put(targets)
  mask = jax.device_put(causal)

  t_compile = time.perf_counter()
  loss0, params, m, v = train_step(params, m, v, tok, tgt, mask, jnp.int32(1))
  jax.block_until_ready(loss0)
  compile_s = time.perf_counter() - t_compile

  for s in (2, 3):
    loss_end, params, m, v = train_step(params, m, v, tok, tgt, mask, jnp.int32(s))
  jax.block_until_ready(loss_end)

  times_ms = []
  for s in range(4, 4 + steps):
    t0 = time.perf_counter()
    loss_end, params, m, v = train_step(params, m, v, tok, tgt, mask, jnp.int32(s))
    jax.block_until_ready(loss_end)
    times_ms.append((time.perf_counter() - t0) * 1e3)

  return _summarize_times(times_ms, compile_s, float(loss0), float(loss_end))


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run_benchmark(
  config_names: list[str],
  *,
  device: str = "cuda",
  steps: int = 20,
  include_host_adam: bool = True,
  include_mapped: bool = True,
) -> dict:
  pk_backend = "cuda" if device == "cuda" else "multicore"
  prof = str(par.target_profile(pk_backend))
  print(f"Parakeet TargetProfile ({pk_backend}): {prof}", flush=True)
  results = {"device": device, "profile": prof, "steps": steps, "configs": {}}

  for key in config_names:
    cfg = DEFAULT_CONFIGS[key]
    rng = np.random.default_rng(42)
    initial = init_gpt(cfg, seed=0)
    tokens = rng.integers(0, cfg.vocab, size=(cfg.batch, cfg.context), dtype=np.int64)
    targets = rng.integers(0, cfg.vocab, size=(cfg.batch, cfg.context), dtype=np.int64)
    causal = np.tril(np.ones((cfg.context, cfg.context), dtype=bool))

    runners = [
      (
        "parakeet_batched_fp32",
        lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_parakeet(
          c, ini, tk, tg, cs, variant="batched", backend=pk_backend, tensor_cores=False, steps=steps
        ),
      ),
    ]
    if device == "cuda":
      runners.append(
        (
          "parakeet_batched_tf32",
          lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_parakeet(
            c, ini, tk, tg, cs, variant="batched", backend="cuda", tensor_cores=True, steps=steps
          ),
        )
      )
    if include_mapped:
      runners.append(
        (
          "parakeet_mapped_fp32",
          lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_parakeet(
            c, ini, tk, tg, cs, variant="mapped", backend=pk_backend, tensor_cores=False, steps=steps
          ),
        )
      )
      if device == "cuda":
        runners.append(
          (
            "parakeet_mapped_tf32",
            lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_parakeet(
              c, ini, tk, tg, cs, variant="mapped", backend="cuda", tensor_cores=True, steps=steps
            ),
          )
        )
    if include_host_adam:
      runners.append(
        (
          "parakeet_host_adam_fp32",
          lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_parakeet(
            c, ini, tk, tg, cs, variant="host_adam", backend=pk_backend, tensor_cores=False, steps=steps
          ),
        )
      )

    runners.extend(
      [
        (
          "torch_eager_fp32",
          lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_torch(
            c, ini, tk, tg, cs, device_name=device, compiled=False, tf32=False, steps=steps
          ),
        ),
        (
          "torch_compile_fp32",
          lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_torch(
            c, ini, tk, tg, cs, device_name=device, compiled=True, tf32=False, steps=steps
          ),
        ),
      ]
    )
    if device == "cuda":
      runners.extend(
        [
          (
            "torch_compile_tf32",
            lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_torch(
              c, ini, tk, tg, cs, device_name="cuda", compiled=True, tf32=True, steps=steps
            ),
          ),
          (
            "torch_sdpa_compile_tf32",
            lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_torch(
              c, ini, tk, tg, cs, device_name="cuda", compiled=True, tf32=True, use_sdpa=True, steps=steps
            ),
          ),
        ]
      )
    runners.append(
      (
        "jax_jit_fp32",
        lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_jax(
          c, ini, tk, tg, cs, tf32=False, steps=steps
        ),
      )
    )
    if device == "cuda":
      runners.append(
        (
          "jax_jit_tf32",
          lambda c=cfg, ini=initial, tk=tokens, tg=targets, cs=causal: bench_jax(
            c, ini, tk, tg, cs, tf32=True, steps=steps
          ),
        )
      )

    print(f"\n=== {cfg.name} ===", flush=True)
    row = {}
    ref_loss0 = None
    for label, fn in runners:
      try:
        res = fn()
        if ref_loss0 is None:
          ref_loss0 = res["loss0"]
        elif abs(res["loss0"] - ref_loss0) > 1e-2:
          res["loss_mismatch_vs_ref"] = round(abs(res["loss0"] - ref_loss0), 5)
        row[label] = res
        print(
          f"  {label:26s}: {res['median_ms']:7.3f} ms (IQR {res['iqr_ms']:5.3f}) "
          f"compile={res['compile_s']:6.2f}s  loss {res['loss0']:.4f} -> {res['loss_end']:.4f}",
          flush=True,
        )
      except Exception as exc:
        row[label] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  {label:26s}: ERROR {type(exc).__name__}: {exc}", flush=True)
    results["configs"][cfg.name] = row

  return results


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
  parser.add_argument("--configs", nargs="+", choices=tuple(DEFAULT_CONFIGS), default=list(DEFAULT_CONFIGS))
  parser.add_argument("--steps", type=int, default=20)
  parser.add_argument("--no-mapped", action="store_true", help="Skip the per-sequence par.map variant")
  parser.add_argument("--no-host-adam", action="store_true", help="Skip the plain-Python Adam variant")
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args(argv)

  results = run_benchmark(
    args.configs,
    device=args.device,
    steps=args.steps,
    include_mapped=not args.no_mapped,
    include_host_adam=not args.no_host_adam,
  )
  encoded = json.dumps(results, indent=2) + "\n"
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(f"\nWrote {args.output}", flush=True)
  print("\nSUMMARY_JSON:" + json.dumps(results), flush=True)


if __name__ == "__main__":
  main()
