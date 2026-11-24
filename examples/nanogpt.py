"""A character-level GPT trained with parakeet.value_and_grad.

The model is a function over a tuple of parameter arrays: token and position
embeddings, a stack of pre-norm transformer blocks (causal self-attention and
an MLP), and a linear head.  ``value_and_grad`` differentiates the loss of a
batch with respect to all of them, and plain Python applies Adam.
"""

import time

import numpy as np

from parakeet import dsl, value_and_grad

N_HEAD = 2


def layer_norm(x, gain, bias):
  mu = np.mean(x, axis=1, keepdims=True)
  var = np.mean((x - mu) ** 2, axis=1, keepdims=True)
  return (x - mu) / np.sqrt(var + 1e-5) * gain + bias


def softmax(x):
  e = np.exp(x - np.max(x, axis=1, keepdims=True))
  return e / np.sum(e, axis=1, keepdims=True)


# --8<-- [start:gpt_attention]
def attention(x, wq, wk, wv, wo, causal):
  """Causal self-attention over one sequence: x is (T, C)."""
  head_width = x.shape[1] // N_HEAD
  q, k, v = x @ wq, x @ wk, x @ wv
  out = x * 0.0
  for h in range(N_HEAD):
    lo, hi = h * head_width, (h + 1) * head_width
    scores = q[:, lo:hi] @ k[:, lo:hi].T / np.sqrt(head_width)
    weights = softmax(np.where(causal, scores, -1e9))
    # Add this head’s contribution to the output embedding.
    out = out + (weights @ v[:, lo:hi]) @ wo[lo:hi]
  return out


# --8<-- [end:gpt_attention]


def block(x, p, causal):
  g1, b1, wq, wk, wv, wo, g2, b2, w1, c1, w2, c2 = p
  x = x + attention(layer_norm(x, g1, b1), wq, wk, wv, wo, causal)
  h = np.tanh(layer_norm(x, g2, b2) @ w1 + c1)
  return x + h @ w2 + c2


def sequence_loss(params, tokens, targets, causal):
  wte, wpe, blocks, gf, bf, head = params
  x = wte[tokens] + wpe
  for p in blocks:
    x = block(x, p, causal)
  logits = layer_norm(x, gf, bf) @ head
  z = logits - np.max(logits, axis=1, keepdims=True)
  logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
  return -np.mean(logp[np.arange(len(targets)), targets])


# --8<-- [start:gpt_batch]
def loss(params, tokens, targets, causal):
  def loss_for_sequence(input_ids, next_ids):
    return sequence_loss(params, input_ids, next_ids, causal)

  per_sequence = dsl.map(loss_for_sequence, tokens, targets)
  return np.mean(per_sequence)


loss_and_grad = value_and_grad(loss)
# --8<-- [end:gpt_batch]


def init(vocab, context, width, layers, rng):
  def w(*shape):
    return rng.normal(0.0, 0.02, shape)

  def one_block():
    ones, zeros = np.ones(width), np.zeros(width)
    return (
      ones, zeros, w(width, width), w(width, width), w(width, width), w(width, width),
      ones.copy(), zeros.copy(), w(width, 4 * width), np.zeros(4 * width), w(4 * width, width), zeros.copy(),
    )  # fmt: skip

  blocks = tuple(one_block() for _ in range(layers))
  return (w(vocab, width), w(context, width), blocks, np.ones(width), np.zeros(width), w(width, vocab))


def tree_map(f, *trees):
  if isinstance(trees[0], tuple):
    return tuple(tree_map(f, *parts) for parts in zip(*trees))
  return f(*trees)


def adam(params, grads, m, v, step, lr=3e-3, b1=0.9, b2=0.99):
  m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
  v = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
  scale = lr * np.sqrt(1 - b2**step) / (1 - b1**step)
  params = tree_map(lambda p, m, v: p - scale * m / (np.sqrt(v) + 1e-8), params, m, v)
  return params, m, v


# --8<-- [start:gpt_update]
def train_step(params, m, v, step, tokens, targets, causal):
  value, grads = loss_and_grad(params, tokens, targets, causal)
  params, m, v = adam(params, grads, m, v, step)
  return value, params, m, v


# --8<-- [end:gpt_update]


TEXT = (
  "parakeet compiles numerical python. it fuses array expressions into loops, "
  "runs them on every core, and now differentiates them too. "
) * 8


def main(steps=60, batch=16, context=32, width=32, layers=2, seed=0, verbose=False):
  rng = np.random.default_rng(seed)
  chars = sorted(set(TEXT))
  data = np.array([chars.index(c) for c in TEXT])
  params = init(len(chars), context, width, layers, rng)
  m = tree_map(np.zeros_like, params)
  v = tree_map(np.zeros_like, params)
  causal = np.tril(np.ones((context, context))) > 0
  first = last = None
  start = time.perf_counter()
  for step in range(1, steps + 1):
    at = rng.integers(0, len(data) - context - 1, batch)
    tokens = np.stack([data[i : i + context] for i in at])
    targets = np.stack([data[i + 1 : i + context + 1] for i in at])
    value, params, m, v = train_step(params, m, v, step, tokens, targets, causal)
    first = value if first is None else first
    last = value
    if verbose and (step == 1 or step % 20 == 0):
      print(f"step {step:3d}  loss {value:.3f}  ({time.perf_counter() - start:.1f}s)")
  assert last < 0.8 * first, (first, last)
  return first, last


if __name__ == "__main__":
  main(verbose=True)
