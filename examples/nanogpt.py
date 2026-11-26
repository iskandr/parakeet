# --8<-- [start:gpt_setup]
"""A character-level GPT trained with parakeet.value_and_grad.

The model is a small hierarchy of ``@dataclass`` modules (``LayerNorm``,
``Block``, ``NanoGPT``) holding plain NumPy arrays. ``par.value_and_grad``
differentiates the batch loss with respect to the entire model dataclass, and
Adam updates each parameter array in plain Python.
"""

import time
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

import parakeet as par

N_HEAD = 2


def softmax(x):
  e = np.exp(x - np.max(x, axis=1, keepdims=True))
  return e / np.sum(e, axis=1, keepdims=True)


@dataclass
class LayerNorm:
  gain: np.ndarray
  bias: np.ndarray

  def __call__(self, x):
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * self.gain + self.bias


# --8<-- [end:gpt_setup]


# --8<-- [start:gpt_attention]
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

  def attention(self, x, causal):
    """Causal self-attention over one sequence: x is (T, C)."""
    head_width = x.shape[1] // N_HEAD
    q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
    out = np.zeros_like(x)
    for h in range(N_HEAD):
      lo, hi = h * head_width, (h + 1) * head_width
      scores = q[:, lo:hi] @ k[:, lo:hi].T / np.sqrt(head_width)
      weights = softmax(np.where(causal, scores, -1e9))
      out = out + (weights @ v[:, lo:hi]) @ self.wo[lo:hi]
    return out

  def __call__(self, x, causal):
    x = x + self.attention(self.ln1(x), causal)
    h = np.tanh(self.ln2(x) @ self.w1 + self.b1)
    return x + h @ self.w2 + self.b2


# --8<-- [end:gpt_attention]


# --8<-- [start:gpt_loss]
@dataclass
class NanoGPT:
  wte: np.ndarray
  wpe: np.ndarray
  blocks: tuple
  ln_f: LayerNorm
  head: np.ndarray

  def sequence_loss(self, tokens, targets, causal):
    x = self.wte[tokens] + self.wpe
    for blk in self.blocks:
      x = blk(x, causal)
    logits = self.ln_f(x) @ self.head
    z = logits - np.max(logits, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return -np.mean(logp[np.arange(len(targets)), targets])


def loss(model, tokens, targets, causal):
  per_sequence = par.map(lambda inp, tgt: model.sequence_loss(inp, tgt, causal), tokens, targets)
  return np.mean(per_sequence)


loss_and_grad = par.value_and_grad(loss)
# --8<-- [end:gpt_loss]


# --8<-- [start:gpt_params]
def init(vocab, context, width, layers, rng):
  def w(*shape):
    return rng.normal(0.0, 0.02, shape)

  def norm():
    return LayerNorm(np.ones(width), np.zeros(width))

  def one_block():
    return Block(
      ln1=norm(),
      wq=w(width, width),
      wk=w(width, width),
      wv=w(width, width),
      wo=w(width, width),
      ln2=norm(),
      w1=w(width, 4 * width),
      b1=np.zeros(4 * width),
      w2=w(4 * width, width),
      b2=np.zeros(width),
    )

  return NanoGPT(
    wte=w(vocab, width),
    wpe=w(context, width),
    blocks=tuple(one_block() for _ in range(layers)),
    ln_f=norm(),
    head=w(width, vocab),
  )


# --8<-- [end:gpt_params]


# --8<-- [start:gpt_update]
def tree_map(f, *trees):
  first = trees[0]
  if isinstance(first, tuple):
    return tuple(tree_map(f, *parts) for parts in zip(*trees))
  if is_dataclass(first):
    return type(first)(*(tree_map(f, *(getattr(t, fl.name) for t in trees)) for fl in fields(first)))
  return f(*trees)


def adam(model, grads, m, v, step, lr=3e-3, b1=0.9, b2=0.99):
  m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
  v = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
  scale = lr * np.sqrt(1 - b2**step) / (1 - b1**step)
  model = tree_map(lambda p, m, v: p - scale * m / (np.sqrt(v) + 1e-8), model, m, v)
  return model, m, v


def train_step(model, m, v, step, tokens, targets, causal):
  value, grads = loss_and_grad(model, tokens, targets, causal)
  model, m, v = adam(model, grads, m, v, step)
  return value, model, m, v


# --8<-- [end:gpt_update]


# --8<-- [start:gpt_train]
TEXT = (
  "parakeet compiles numerical python. it fuses array expressions into loops, "
  "runs them on every core, and now differentiates them too. "
) * 8


def main(steps=60, batch=16, context=32, width=32, layers=2, seed=0, verbose=False):
  rng = np.random.default_rng(seed)
  chars = sorted(set(TEXT))
  data = np.array([chars.index(c) for c in TEXT])
  model = init(len(chars), context, width, layers, rng)
  m = tree_map(np.zeros_like, model)
  v = tree_map(np.zeros_like, model)
  causal = np.tril(np.ones((context, context))) > 0
  first = last = None
  start = time.perf_counter()
  for step in range(1, steps + 1):
    at = rng.integers(0, len(data) - context - 1, batch)
    tokens = np.stack([data[i : i + context] for i in at])
    targets = np.stack([data[i + 1 : i + context + 1] for i in at])
    value, model, m, v = train_step(model, m, v, step, tokens, targets, causal)
    first = value if first is None else first
    last = value
    if verbose and (step == 1 or step % 20 == 0):
      print(f"step {step:3d}  loss {value:.3f}  ({time.perf_counter() - start:.1f}s)")
  assert last < 0.8 * first, (first, last)
  return first, last


if __name__ == "__main__":
  main(verbose=True)
# --8<-- [end:gpt_train]
