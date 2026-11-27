"""Minimal, composable neural-network primitives and Pythonic modules.

Every functional helper is plain NumPy so Parakeet's algebraic passes
(monoidal FlashAttention, cut cross-entropy, warp-row LayerNorm, and fused
elementwise loops) trigger automatically.  Every :class:`Module` subclass is a
normal Python object with a standard ``__init__`` that crosses ``@par.jit``,
``par.value_and_grad``, ``par.to_device``, and ``par.tree_map`` with zero
``PyObject`` overhead.
"""

from __future__ import annotations

import numpy as np

from .tree import tree_array_leaves

# ---------------------------------------------------------------------------
# Functional primitives
# ---------------------------------------------------------------------------


def relu(x):
  return np.maximum(x, 0)


def gelu(x):
  """Tanh-approximated GELU activation."""
  c = np.float32(0.7978845608) if x.dtype == np.float32 else 0.7978845608
  k = np.float32(0.044715) if x.dtype == np.float32 else 0.044715
  half = np.float32(0.5) if x.dtype == np.float32 else 0.5
  one = np.float32(1.0) if x.dtype == np.float32 else 1.0
  return half * x * (one + np.tanh(c * (x + k * x * x * x)))


def silu(x):
  """SiLU / Swish activation: ``x / (1 + exp(-x))``."""
  one = np.float32(1.0) if x.dtype == np.float32 else 1.0
  return x / (one + np.exp(-x))


def softmax(x, axis: int = -1):
  shifted = x - np.max(x, axis=axis, keepdims=True)
  exps = np.exp(shifted)
  return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x, axis: int = -1):
  shifted = x - np.max(x, axis=axis, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def layer_norm(x, gain, bias, eps=1e-5, axis: int = -1):
  mu = np.mean(x, axis=axis, keepdims=True)
  diff = x - mu
  var = np.mean(diff * diff, axis=axis, keepdims=True)
  eps_v = np.float32(eps) if x.dtype == np.float32 else eps
  return diff / np.sqrt(var + eps_v) * gain + bias


def rms_norm(x, gain, eps=1e-5, axis: int = -1):
  ms = np.mean(x * x, axis=axis, keepdims=True)
  eps_v = np.float32(eps) if x.dtype == np.float32 else eps
  return x / np.sqrt(ms + eps_v) * gain


def causal_mask(seq_len: int):
  """Lower-triangular boolean mask of shape ``(seq_len, seq_len)``."""
  idx = np.arange(seq_len)
  return idx[:, None] >= idx[None, :]


def scaled_dot_product_attention(q, k, v, *, causal: bool = False, mask=None):
  """Scaled dot-product attention over ``(..., T_q, D)``, ``(..., T_kv, D)``, ``(..., T_kv, D_v)``."""
  head_dim = q.shape[-1]
  scale = np.float32(1.0 / np.sqrt(np.float32(head_dim))) if q.dtype == np.float32 else (1.0 / np.sqrt(head_dim))
  scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
  neg_large = np.float32(-1e9) if q.dtype == np.float32 else -1e9
  if mask is not None:
    scores = np.where(mask, scores, neg_large)
  elif causal:
    scores = np.where(causal_mask(q.shape[-2]), scores, neg_large)
  return np.matmul(softmax(scores, axis=-1), v)


def cross_entropy(logits, targets):
  """Mean cross-entropy loss for ``(N, V)`` or ``(B, T, V)`` logits and integer targets."""
  if logits.ndim == 3:
    b, t, v = logits.shape
    flat_logits = logits.reshape((b * t, v))
    flat_targets = targets.reshape((b * t,))
  else:
    flat_logits = logits
    flat_targets = targets
  shifted = flat_logits - np.max(flat_logits, axis=1, keepdims=True)
  log_denom = np.log(np.sum(np.exp(shifted), axis=1))
  picked = shifted[np.arange(flat_logits.shape[0]), flat_targets]
  return -np.mean(picked - log_denom)


# ---------------------------------------------------------------------------
# Pythonic Module hierarchy
# ---------------------------------------------------------------------------


def _default_rng(rng):
  return np.random.default_rng(0) if rng is None else rng


class Module:
  """Base class for trainable Parakeet modules with ordinary Python ``__init__``."""

  __parakeet_record__ = True

  def parameters(self) -> list:
    """Return all parameter arrays in deterministic depth-first order."""
    return tree_array_leaves(self)

  def __repr__(self) -> str:
    parts = [
      f"{k}={type(v).__name__}{tuple(v.shape)}" if isinstance(v, np.ndarray) else f"{k}={v!r}"
      for k, v in self.__dict__.items()
    ]
    return f"{type(self).__name__}({', '.join(parts)})"


class Linear(Module):
  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    rng=None,
    dtype=np.float32,
    std: float | None = None,
  ):
    g = _default_rng(rng)
    scale = (1.0 / np.sqrt(in_features)) if std is None else float(std)
    self.weight = g.normal(0.0, scale, (in_features, out_features)).astype(dtype)
    self.bias = np.zeros(out_features, dtype=dtype) if bias else None

  def __call__(self, x):
    y = x @ self.weight
    if self.bias is not None:
      return y + self.bias
    return y


class LayerNorm(Module):
  def __init__(self, width: int, *, eps: float = 1e-5, dtype=np.float32):
    self.gain = np.ones(width, dtype=dtype)
    self.bias = np.zeros(width, dtype=dtype)
    self.eps = float(eps)

  def __call__(self, x):
    return layer_norm(x, self.gain, self.bias, eps=self.eps, axis=-1)


class Embedding(Module):
  def __init__(self, num_embeddings: int, width: int, *, rng=None, dtype=np.float32, std: float = 0.02):
    g = _default_rng(rng)
    self.weight = g.normal(0.0, std, (num_embeddings, width)).astype(dtype)

  def __call__(self, indices):
    if indices.ndim == 2:
      b, t = indices.shape
      return self.weight[indices.reshape((b * t,))].reshape((b, t, self.weight.shape[1]))
    return self.weight[indices]


class MLP(Module):
  def __init__(
    self,
    width: int,
    hidden: int | None = None,
    *,
    activation: str = "tanh",
    rng=None,
    dtype=np.float32,
    std: float = 0.02,
  ):
    g = _default_rng(rng)
    h = 4 * width if hidden is None else hidden
    self.fc1 = Linear(width, h, rng=g, dtype=dtype, std=std)
    self.fc2 = Linear(h, width, rng=g, dtype=dtype, std=std)
    self.activation = activation

  def __call__(self, x):
    h = self.fc1(x)
    if self.activation == "tanh":
      h = np.tanh(h)
    elif self.activation == "gelu":
      h = gelu(h)
    elif self.activation == "silu":
      h = silu(h)
    else:
      h = relu(h)
    return self.fc2(h)


class CausalSelfAttention(Module):
  def __init__(self, width: int, heads: int, *, rng=None, dtype=np.float32, std: float = 0.02):
    if width % heads != 0:
      raise ValueError(f"width ({width}) must be divisible by heads ({heads})")
    g = _default_rng(rng)
    self.heads = int(heads)
    self.wq = Linear(width, width, bias=False, rng=g, dtype=dtype, std=std)
    self.wk = Linear(width, width, bias=False, rng=g, dtype=dtype, std=std)
    self.wv = Linear(width, width, bias=False, rng=g, dtype=dtype, std=std)
    self.wo = Linear(width, width, bias=False, rng=g, dtype=dtype, std=std)

  def __call__(self, x, mask=None):
    if x.ndim == 3:
      b, t, c = x.shape
      d = c // self.heads
      q = self.wq(x).reshape((b, t, self.heads, d)).transpose((0, 2, 1, 3))
      k = self.wk(x).reshape((b, t, self.heads, d)).transpose((0, 2, 1, 3))
      v = self.wv(x).reshape((b, t, self.heads, d)).transpose((0, 2, 1, 3))
      out = scaled_dot_product_attention(q, k, v, causal=(mask is None), mask=mask)
      return self.wo(out.transpose((0, 2, 1, 3)).reshape((b, t, c)))
    t, c = x.shape
    d = c // self.heads
    q = self.wq(x).reshape((t, self.heads, d)).transpose((1, 0, 2))
    k = self.wk(x).reshape((t, self.heads, d)).transpose((1, 0, 2))
    v = self.wv(x).reshape((t, self.heads, d)).transpose((1, 0, 2))
    out = scaled_dot_product_attention(q, k, v, causal=(mask is None), mask=mask)
    return self.wo(out.transpose((1, 0, 2)).reshape((t, c)))


class TransformerBlock(Module):
  def __init__(
    self,
    width: int,
    heads: int,
    *,
    mlp_ratio: int = 4,
    activation: str = "tanh",
    rng=None,
    dtype=np.float32,
    std: float = 0.02,
  ):
    g = _default_rng(rng)
    self.ln1 = LayerNorm(width, dtype=dtype)
    self.attn = CausalSelfAttention(width, heads, rng=g, dtype=dtype, std=std)
    self.ln2 = LayerNorm(width, dtype=dtype)
    self.mlp = MLP(width, mlp_ratio * width, activation=activation, rng=g, dtype=dtype, std=std)

  def __call__(self, x, mask=None):
    x = x + self.attn(self.ln1(x), mask=mask)
    return x + self.mlp(self.ln2(x))
