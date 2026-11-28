"""Minimal, JIT-compilable optimizers for Parakeet models and parameter trees.

Both functional update steps (:func:`sgd_step`, :func:`adam_step`) and a
stateful :class:`Adam` helper are provided. Functional steps work identically
in plain Python and inside ``@par.jit`` via :func:`parakeet.tree.tree_map_floating`,
leaving static module attributes and non-floating buffers (such as boolean
masks or integer lookup tables) untouched while preserving each parameter's
exact floating-point dtype (e.g. ``float32`` stays ``float32``).
"""

from __future__ import annotations

import numpy as np

from .tree import tree_map_floating


def sgd_step(params, grads, lr=1e-2):
  """One SGD update step over floating parameter leaves (Python or inside ``@par.jit``)."""
  return tree_map_floating(lambda p, g: (p - lr * g).astype(p.dtype), params, grads)


def adam_init(params):
  """Create zero-initialized ``(m, v)`` moment trees matching floating leaves of ``params``."""
  return tree_map_floating(np.zeros_like, params), tree_map_floating(np.zeros_like, params)


def adam_step(params, grads, m, v, step, lr=3e-3, b1=0.9, b2=0.99, eps=1e-8):
  """One Adam update step returning ``(new_params, new_m, new_v)``.

  Preserves each parameter leaf's dtype (``float32`` stays ``float32``) and
  works both in plain Python and inside ``@par.jit`` (e.g. fused with
  ``par.value_and_grad`` for zero-round-trip GPU training steps).
  """
  m_next = tree_map_floating(lambda mi, g: (b1 * mi + (1.0 - b1) * g).astype(mi.dtype), m, grads)
  v_next = tree_map_floating(lambda vi, g: (b2 * vi + (1.0 - b2) * g * g).astype(vi.dtype), v, grads)
  scale = lr * np.sqrt(1.0 - b2**step) / (1.0 - b1**step)
  params_next = tree_map_floating(
    lambda p, mi, vi: p - (scale * mi / (np.sqrt(vi) + eps)).astype(p.dtype),
    params,
    m_next,
    v_next,
  )
  return params_next, m_next, v_next


class Adam:
  """Stateful Pythonic Adam optimizer wrapping :func:`adam_step`."""

  def __init__(self, params, lr: float = 3e-3, betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8):
    self.lr = lr
    self.b1, self.b2 = betas
    self.eps = eps
    self.step_count = 0
    self.m, self.v = adam_init(params)

  def update(self, params, grads):
    """Apply one Adam update and return the updated parameter tree."""
    self.step_count += 1
    params, self.m, self.v = adam_step(
      params,
      grads,
      self.m,
      self.v,
      self.step_count,
      lr=self.lr,
      b1=self.b1,
      b2=self.b2,
      eps=self.eps,
    )
    return params
