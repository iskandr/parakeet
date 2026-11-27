"""Tests for par.nn, par.optim, par.tree_map, and par.Alphabet / par.Vocab."""

from __future__ import annotations

import numpy as np

import parakeet as par


def test_alphabet_and_vocab_python_and_jit():
  dna = par.Alphabet("ACGT")
  gc = par.Alphabet.mask("GC")

  assert len(dna) == 4
  assert dna.code("A") == 0 and dna.code("T") == 3
  codes = dna.encode("ACGTacgtN")
  np.testing.assert_array_equal(codes, np.array([0, 1, 2, 3, 0, 1, 2, 3, 4], dtype=np.uint8))
  assert dna.decode(codes) == "ACGTACGT?"

  # Works on list[str] directly inside @par.jit with zero IntEnum/dict boilerplate
  @par.jit
  def gc_and_codes(reads):
    return par.map(lambda seq: (dna[seq], np.mean(gc[seq].astype(np.float32))), reads)

  reads = ["ACGTACGT", "GGGGCCCC", "AAAATTTT"]
  out_codes, out_gc = gc_and_codes(reads)
  np.testing.assert_array_equal(out_codes[0], np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8))
  np.testing.assert_allclose(out_gc, np.array([0.5, 1.0, 0.0], dtype=np.float32))

  # RaggedArray encoding
  rag = dna.encode(["ACG", "TTTTA"])
  assert isinstance(rag, par.RaggedArray)
  assert len(rag) == 2
  np.testing.assert_array_equal(rag[0], np.array([0, 1, 2], dtype=np.uint8))
  np.testing.assert_array_equal(rag[1], np.array([3, 3, 3, 3, 0], dtype=np.uint8))

  # Vocab roundtrip (Python + inside @par.jit)
  text = "parakeet compiles python!"
  vocab = par.Vocab.from_text(text)
  ids = vocab.encode(text)
  assert vocab.decode(ids) == text

  @par.jit
  def jit_encode(raw_bytes):
    return vocab[raw_bytes]

  np.testing.assert_array_equal(jit_encode(text.encode("ascii")), ids)


def test_nn_module_normal_init_autodiff_and_compiled_adam():
  class TinyGPT(par.nn.Module):
    def __init__(self, vocab: int, context: int, width: int, heads: int, layers: int, rng):
      self.wte = par.nn.Embedding(vocab, width, rng=rng)
      self.wpe = rng.normal(0.0, 0.02, (context, width)).astype(np.float32)
      self.blocks = tuple(par.nn.TransformerBlock(width, heads, rng=rng) for _ in range(layers))
      self.ln_f = par.nn.LayerNorm(width)
      self.head = par.nn.Linear(width, vocab, bias=False, rng=rng, std=0.02)

    def __call__(self, tokens, mask=None):
      x = self.wte(tokens) + self.wpe
      for blk in self.blocks:
        x = blk(x, mask=mask)
      return self.head(self.ln_f(x))

  rng = np.random.default_rng(0)
  model = TinyGPT(vocab=32, context=16, width=32, heads=2, layers=2, rng=rng)
  assert len(model.parameters()) > 10

  tokens = rng.integers(0, 32, size=(4, 16), dtype=np.int64)
  targets = rng.integers(0, 32, size=(4, 16), dtype=np.int64)
  mask = par.nn.causal_mask(16)

  def loss_fn(m, tok, tgt, cas):
    return par.nn.cross_entropy(m(tok, mask=cas), tgt)

  vg = par.value_and_grad(loss_fn)

  # 1. Host Python Adam via par.optim.Adam
  opt = par.optim.Adam(model, lr=3e-3)
  m_host = par.tree_map_arrays(np.copy, model)
  losses_host = []
  for _ in range(4):
    lv, grads = vg(m_host, tokens, targets, mask)
    losses_host.append(float(lv))
    m_host = opt.update(m_host, grads)

  # 2. Compiled Adam step via par.optim.adam_step inside @par.jit
  @par.jit
  def compiled_step(m, st_m, st_v, step, tok, tgt, cas):
    lv, grads = vg(m, tok, tgt, cas)
    m, st_m, st_v = par.optim.adam_step(m, grads, st_m, st_v, step, lr=3e-3)
    return lv, m, st_m, st_v

  m_jit = par.tree_map_arrays(np.copy, model)
  st_m, st_v = par.optim.adam_init(m_jit)
  losses_jit = []
  for s in range(1, 5):
    lv, m_jit, st_m, st_v = compiled_step(m_jit, st_m, st_v, np.float32(s), tokens, targets, mask)
    losses_jit.append(float(lv))

  assert losses_host[-1] < losses_host[0]
  np.testing.assert_allclose(losses_jit, losses_host, rtol=1e-4, atol=1e-4)
