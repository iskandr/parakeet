# --8<-- [start:gpt_functional_primitives]
"""Minimalist functional nanoGPT written from scratch without ``par.nn`` or ``par.optim``.

Parameters live in plain dicts and tuples of NumPy arrays. Every neural-network
building block (layer normalization, softmax, multi-head causal attention,
cross-entropy, and Adam) is written inline in a few lines of NumPy so there are
no hidden abstractions.

Compare with the Pythonic ``par.nn.Module`` + ``par.optim.Adam`` version in
``examples/nanogpt.py``.
"""

from __future__ import annotations

import time

import numpy as np

import parakeet as par

N_HEAD = 2


def layer_norm(x, gain, bias, eps=1e-5):
  mu = np.mean(x, axis=-1, keepdims=True)
  var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
  return (x - mu) / np.sqrt(var + eps) * gain + bias


def softmax(x):
  shifted = x - np.max(x, axis=-1, keepdims=True)
  exps = np.exp(shifted)
  return exps / np.sum(exps, axis=-1, keepdims=True)


def causal_attention(x, block, mask):
  """Multi-head causal self-attention over ``(B, T, C)``."""
  b, t, c = x.shape
  d = c // N_HEAD
  q = (x @ block["wq"]).reshape((b, t, N_HEAD, d)).transpose((0, 2, 1, 3))
  k = (x @ block["wk"]).reshape((b, t, N_HEAD, d)).transpose((0, 2, 1, 3))
  v = (x @ block["wv"]).reshape((b, t, N_HEAD, d)).transpose((0, 2, 1, 3))
  scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d)
  weights = softmax(np.where(mask, scores, -1e9))
  out = np.matmul(weights, v).transpose((0, 2, 1, 3)).reshape((b, t, c))
  return out @ block["wo"]


def transformer_block(x, block, mask):
  x = x + causal_attention(layer_norm(x, block["g1"], block["b1"]), block, mask)
  h = np.tanh(layer_norm(x, block["g2"], block["b2"]) @ block["w1"] + block["bf1"])
  return x + h @ block["w2"] + block["bf2"]


# --8<-- [end:gpt_functional_primitives]


# --8<-- [start:gpt_functional_loss]
def loss(params, tokens, targets, mask):
  b, t = tokens.shape
  c = params["wte"].shape[1]
  x = params["wte"][tokens.reshape((b * t,))].reshape((b, t, c)) + params["wpe"]
  for block in params["blocks"]:
    x = transformer_block(x, block, mask)
  logits = (layer_norm(x, params["gf"], params["bf"]) @ params["head"]).reshape((b * t, params["head"].shape[1]))
  shifted = logits - np.max(logits, axis=1, keepdims=True)
  log_denom = np.log(np.sum(np.exp(shifted), axis=1))
  picked = shifted[np.arange(b * t), targets.reshape((b * t,))]
  return -np.mean(picked - log_denom)


loss_and_grad = par.value_and_grad(loss)


def init_params(vocab, context, width, layers, rng):
  w = lambda *shape: rng.normal(0.0, 0.02, shape)  # noqa: E731
  blocks = tuple(
    {
      "g1": np.ones(width),
      "b1": np.zeros(width),
      "wq": w(width, width),
      "wk": w(width, width),
      "wv": w(width, width),
      "wo": w(width, width),
      "g2": np.ones(width),
      "b2": np.zeros(width),
      "w1": w(width, 4 * width),
      "bf1": np.zeros(4 * width),
      "w2": w(4 * width, width),
      "bf2": np.zeros(width),
    }
    for _ in range(layers)
  )
  return {
    "wte": w(vocab, width),
    "wpe": w(context, width),
    "blocks": blocks,
    "gf": np.ones(width),
    "bf": np.zeros(width),
    "head": w(width, vocab),
  }


def adam_step(params, grads, m, v, step, lr=3e-3, b1=0.9, b2=0.99):
  m = par.tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, m, grads)
  v = par.tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * g * g, v, grads)
  scale = lr * np.sqrt(1.0 - b2**step) / (1.0 - b1**step)
  params = par.tree_map(lambda p, mi, vi: p - scale * mi / (np.sqrt(vi) + 1e-8), params, m, v)
  return params, m, v


# --8<-- [end:gpt_functional_loss]


TEXT = (
  "parakeet compiles numerical python. it fuses array expressions into loops, "
  "runs them on every core, and now differentiates them too. "
) * 8


def main(steps=60, batch=16, context=32, width=32, layers=2, seed=0, verbose=False):
  rng = np.random.default_rng(seed)
  vocab = par.Vocab.from_text(TEXT)
  data = vocab.encode(TEXT)

  params = init_params(len(vocab), context, width, layers, rng)
  m = par.tree_map(np.zeros_like, params)
  v = par.tree_map(np.zeros_like, params)
  idx = np.arange(context)
  mask = idx[:, None] >= idx[None, :]

  first = last = None
  start = time.perf_counter()
  for step in range(1, steps + 1):
    starts = rng.integers(0, len(data) - context - 1, batch)
    tokens = np.stack([data[i : i + context] for i in starts])
    targets = np.stack([data[i + 1 : i + context + 1] for i in starts])
    value, grads = loss_and_grad(params, tokens, targets, mask)
    params, m, v = adam_step(params, grads, m, v, step)
    first = value if first is None else first
    last = value
    if verbose and (step == 1 or step % 20 == 0):
      print(f"step {step:3d}  loss {value:.3f}  ({time.perf_counter() - start:.1f}s)")

  assert last < 0.8 * first, (first, last)
  return first, last


if __name__ == "__main__":
  main(verbose=True)
