"""Idiomatic Python objects, methods, structural matching, and polymorphic dispatch in Parakeet.

Run directly to verify exactness (including gradients) and compare speed against NumPy::

    python -m examples.protein_and_layers
"""

import functools
import operator
import time
from dataclasses import dataclass

import numpy as np

import parakeet as par
from parakeet import RaggedArray


# --8<-- [start:protein_and_layers]
@dataclass
class ProteinBackbone:
  """Mutable-style @dataclass with pure methods, @property, and functional field updates."""

  coords: np.ndarray  # (N, 3, 3) for N, CA, C atoms per residue
  residue_ids: np.ndarray  # (N,) int64 metadata (zero-tangent static leaf under AD)
  chain_name: str = "A"

  @property
  def ca_coords(self):
    return self.coords[:, 1, :]

  @property
  def n_residues(self):
    return self.coords.shape[0]

  def centered(self):
    mu = np.mean(self.ca_coords, axis=0, keepdims=True)
    self.coords = self.coords - mu[:, None, :]
    return self

  def rbf_message_passing(self, features, neg_gamma):
    ca = self.ca_coords
    d2 = np.sum((ca[:, None, :] - ca[None, :, :]) ** np.float32(2.0), axis=-1)
    s = d2 * neg_gamma
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)) @ features


@dataclass
class MLPBlock:
  """Callable @dataclass layer with polymorphic dense/ragged dispatch."""

  w1: np.ndarray
  b1: np.ndarray
  w2: np.ndarray
  b2: np.ndarray
  name: str = "mlp"

  def __call__(self, x, *, mask=None, scale=1.0):
    if isinstance(x, RaggedArray):
      h = np.tanh(x.values @ self.w1 + self.b1)
      y = (h @ self.w2 + self.b2) * scale
      return RaggedArray(y, x.offsets)
    h = np.tanh(x @ self.w1 + self.b1)
    y = (h @ self.w2 + self.b2) * scale
    if mask is not None:
      y = np.where(mask[:, None], y, np.float32(0.0))
    return y


def apply_with_kwargs(fn, x, *args, **kwargs):
  return fn(x, *args, **kwargs)


@par.jit
def score_structure(backbone, features, block, neg_gamma, mask=None):
  bb = backbone.centered()
  msgs = bb.rbf_message_passing(features, neg_gamma)
  step = functools.partial(apply_with_kwargs, block, scale=np.float32(0.5))
  out = step(msgs, mask=mask)
  match out:
    case RaggedArray(values=vals):
      return par.reduce(operator.add, vals * vals)
    case _:
      return np.sum(out * out)


# --8<-- [end:protein_and_layers]


def main():
  rng = np.random.default_rng(7)
  n_res, feat_dim, hidden = 256, 32, 64
  coords = rng.standard_normal((n_res, 3, 3)).astype(np.float32) * 5.0
  res_ids = np.arange(n_res, dtype=np.int64)
  feats = rng.standard_normal((n_res, feat_dim)).astype(np.float32) * 0.25
  mask = rng.random(n_res) > 0.1
  neg_gamma = np.float32(-0.05)

  bb = ProteinBackbone(coords, res_ids, chain_name="A")
  block = MLPBlock(
    w1=rng.standard_normal((feat_dim, hidden)).astype(np.float32) * 0.2,
    b1=rng.standard_normal((hidden,)).astype(np.float32) * 0.1,
    w2=rng.standard_normal((hidden, feat_dim)).astype(np.float32) * 0.2,
    b2=rng.standard_normal((feat_dim,)).astype(np.float32) * 0.1,
  )

  # Verify dense path + polymorphic RaggedArray path
  val_pk = score_structure(bb, feats, block, neg_gamma, mask)
  val_ref = score_structure(bb, feats, block, neg_gamma, mask, _backend="python")
  np.testing.assert_allclose(val_pk, val_ref, rtol=1e-4, atol=1e-4)

  rag_in = RaggedArray.from_lengths(feats[:64], np.array([16, 20, 28], dtype=np.int64))
  rag_out = par.jit(lambda blk, r: blk(r, scale=np.float32(0.5)))(block, rag_in)
  rag_ref = block(rag_in, scale=np.float32(0.5))
  np.testing.assert_allclose(rag_out.values, rag_ref.values, rtol=1e-5, atol=1e-5)
  np.testing.assert_array_equal(rag_out.offsets, rag_ref.offsets)

  # Verify reverse-mode gradient through non-frozen dataclass methods + discrete metadata
  small_bb = ProteinBackbone(coords[:16].astype(np.float64), res_ids[:16], chain_name="A")
  small_f = feats[:16].astype(np.float64)
  small_blk = MLPBlock(
    block.w1.astype(np.float64),
    block.b1.astype(np.float64),
    block.w2.astype(np.float64),
    block.b2.astype(np.float64),
  )

  def small_loss(b_obj, f_arr):
    c = b_obj.centered()
    m = c.rbf_message_passing(f_arr, -0.05)
    return np.sum(small_blk(m) ** 2)

  g_bb = par.grad(small_loss, 0)(small_bb, small_f)
  assert g_bb.chain_name == "A"
  eps = 1e-6
  c_p = small_bb.coords.copy()
  c_m = small_bb.coords.copy()
  c_p[0, 1, 0] += eps
  c_m[0, 1, 0] -= eps
  fd = (
    small_loss(ProteinBackbone(c_p, res_ids[:16]), small_f) - small_loss(ProteinBackbone(c_m, res_ids[:16]), small_f)
  ) / (2 * eps)
  np.testing.assert_allclose(g_bb.coords[0, 1, 0], fd, rtol=1e-4, atol=1e-5)

  # Quick CPU timing comparison
  score_structure(bb, feats, block, neg_gamma, mask)
  t0 = time.perf_counter()
  for _ in range(5):
    score_structure(bb, feats, block, neg_gamma, mask)
  t_pk = (time.perf_counter() - t0) / 5 * 1e3

  t0 = time.perf_counter()
  for _ in range(5):
    score_structure(bb, feats, block, neg_gamma, mask, _backend="python")
  t_py = (time.perf_counter() - t0) / 5 * 1e3

  print(
    f"ProteinBackbone + MLPBlock ({n_res} residues): Parakeet={t_pk:.2f} ms vs NumPy={t_py:.2f} ms ({t_py / t_pk:.1f}x)"
  )


if __name__ == "__main__":
  main()
