# --8<-- [start:gpt_model]
"""Character-level nanoGPT using ``par.nn.Module``, ``par.optim.Adam``, and ``par.Vocab``.

``NanoGPT`` is a normal Python class with a standard ``__init__``. Because it
inherits from :class:`par.nn.Module`, Parakeet lowers the model and its methods
into compiled native code with zero ``PyObject`` overhead, differentiates the
loss with ``par.value_and_grad``, and updates every parameter array with
``par.optim.Adam``.

For a zero-helper functional version written with plain dicts and raw NumPy,
see ``examples/nanogpt_functional.py``.
"""

from __future__ import annotations

import time

import numpy as np

import parakeet as par


class NanoGPT(par.nn.Module):
  """Two-head character-level transformer with standard Python ``__init__``."""

  def __init__(self, vocab: int, context: int, width: int, layers: int, *, heads: int = 2, rng=None, dtype=np.float64):
    g = np.random.default_rng(0) if rng is None else rng
    self.wte = par.nn.Embedding(vocab, width, rng=g, dtype=dtype)
    self.wpe = g.normal(0.0, 0.02, (context, width)).astype(dtype)
    self.blocks = tuple(par.nn.TransformerBlock(width, heads, rng=g, dtype=dtype) for _ in range(layers))
    self.ln_f = par.nn.LayerNorm(width, dtype=dtype)
    self.head = par.nn.Linear(width, vocab, bias=False, rng=g, dtype=dtype, std=0.02)

  def __call__(self, tokens, mask=None):
    x = self.wte(tokens) + self.wpe
    for block in self.blocks:
      x = block(x, mask=mask)
    return self.head(self.ln_f(x))


def loss(model, tokens, targets, mask):
  return par.nn.cross_entropy(model(tokens, mask=mask), targets)


loss_and_grad = par.value_and_grad(loss)
# --8<-- [end:gpt_model]


# --8<-- [start:gpt_train]
TEXT = (
  "parakeet compiles numerical python. it fuses array expressions into loops, "
  "runs them on every core, and now differentiates them too. "
) * 8


def sample_batch(data, batch: int, context: int, rng):
  starts = rng.integers(0, len(data) - context - 1, batch)
  tokens = np.stack([data[i : i + context] for i in starts])
  targets = np.stack([data[i + 1 : i + context + 1] for i in starts])
  return tokens, targets


def main(steps=60, batch=16, context=32, width=32, layers=2, heads=2, seed=0, verbose=False):
  rng = np.random.default_rng(seed)
  vocab = par.Vocab.from_text(TEXT)
  data = vocab.encode(TEXT)

  model = NanoGPT(len(vocab), context, width, layers, heads=heads, rng=rng)
  optimizer = par.optim.Adam(model, lr=3e-3, betas=(0.9, 0.99))
  mask = par.nn.causal_mask(context)

  first = last = None
  start = time.perf_counter()
  for step in range(1, steps + 1):
    tokens, targets = sample_batch(data, batch, context, rng)
    value, grads = loss_and_grad(model, tokens, targets, mask)
    model = optimizer.update(model, grads)
    first = value if first is None else first
    last = value
    if verbose and (step == 1 or step % 20 == 0):
      print(f"step {step:3d}  loss {value:.3f}  ({time.perf_counter() - start:.1f}s)")

  assert last < 0.8 * first, (first, last)
  return first, last


if __name__ == "__main__":
  main(verbose=True)
# --8<-- [end:gpt_train]
