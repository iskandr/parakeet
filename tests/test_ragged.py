"""Tests for #47: first-class RaggedArray, segmented operators, varlen FlashAttention, and AD."""

from __future__ import annotations

import dataclasses
import os
from typing import NamedTuple

import numpy as np
import pytest

import parakeet as pk
from parakeet import RaggedArray, dsl


class AtomSoA(NamedTuple):
  xyz: np.ndarray
  charge: np.ndarray


@dataclasses.dataclass(frozen=True)
class ResidueSoA:
  coord: np.ndarray
  bfactor: np.ndarray


# ---------------------------------------------------------------------------
# 1. Construction, Python API, indexing, iteration, nested & SoA RaggedArray
# ---------------------------------------------------------------------------


def test_ragged_python_api_and_boundary():
  r = RaggedArray.from_sequences([[1.0, 2.0, 3.0], [], [4.0], [5.0, 6.0]], dtype=np.float32)
  assert len(r) == 4
  assert r.num_segments == 4
  np.testing.assert_array_equal(r.lengths, np.array([3, 0, 1, 2], dtype=np.int64))
  np.testing.assert_array_equal(r[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
  np.testing.assert_array_equal(r[1], np.array([], dtype=np.float32))
  np.testing.assert_array_equal(r[-1], np.array([5.0, 6.0], dtype=np.float32))
  padded, mask = r.to_padded(fill=-1.0, return_mask=True)
  assert padded.shape == (4, 3)
  assert mask.shape == (4, 3)
  assert padded[1, 0] == -1.0 and not mask[1, 0]

  @pk.jit
  def seg_sums(rag):
    out = np.empty(len(rag), dtype=np.float32)
    for i in range(len(rag)):
      out[i] = np.sum(rag[i])
    return out

  np.testing.assert_allclose(seg_sums(r), np.array([6.0, 0.0, 4.0, 11.0], dtype=np.float32))

  @pk.jit
  def sum_via_iter_and_neg_index(rag):
    total = np.float32(0.0)
    for seg in rag:
      total += np.sum(seg)
    return total + np.sum(rag[-1]), rag.lengths, rag.num_segments

  tot, lens, nseg = sum_via_iter_and_neg_index(r)
  np.testing.assert_allclose(tot, 21.0 + 11.0)
  np.testing.assert_array_equal(lens, np.array([3, 0, 1, 2], dtype=np.int64))
  assert nseg == 4


def test_ragged_empty_nested_and_soa():
  # Empty RaggedArray (0 segments) and all-empty segments
  empty0 = RaggedArray(np.empty(0, dtype=np.float32), np.array([0], dtype=np.int64))
  empty_segs = RaggedArray(np.empty(0, dtype=np.float32), np.array([0, 0, 0], dtype=np.int64))

  @pk.jit
  def reduce_with_init(rag):
    return dsl.segmented_reduce("add", rag, init=np.float32(0.0))

  assert reduce_with_init(empty0).shape == (0,)
  np.testing.assert_array_equal(reduce_with_init(empty_segs), np.zeros(2, dtype=np.float32))

  # 2-level nested RaggedArray (Batch -> Chains -> Residues)
  inner = RaggedArray.from_sequences([[1.0, 2.0], [3.0], [], [4.0, 5.0, 6.0]], dtype=np.float32)
  outer = RaggedArray(inner, np.array([0, 2, 4], dtype=np.int64))

  @pk.jit
  def batch_totals(nested):
    out = np.empty(len(nested), dtype=np.float32)
    for b in range(len(nested)):
      chains = nested[b]
      s = np.float32(0.0)
      for c in range(len(chains)):
        s += np.sum(chains[c])
      out[b] = s
    return out

  np.testing.assert_allclose(batch_totals(outer), np.array([6.0, 15.0], dtype=np.float32))

  # SoA RaggedArray of NamedTuple and frozen dataclass
  xyz = np.arange(15, dtype=np.float32).reshape(5, 3)
  charge = np.array([1.0, -0.5, 0.25, 2.0, -1.0], dtype=np.float32)
  mol = RaggedArray(AtomSoA(xyz, charge), np.array([0, 2, 5], dtype=np.int64))

  @pk.jit
  def net_charges(m):
    out = np.empty(len(m), dtype=np.float32)
    for i in range(len(m)):
      atoms = m[i]
      out[i] = np.sum(atoms.charge) + atoms.xyz[0, 0]
    return out

  np.testing.assert_allclose(net_charges(mol), np.array([0.5 + 0.0, 1.25 + 6.0], dtype=np.float32))

  res = RaggedArray(ResidueSoA(xyz, charge), np.array([0, 3, 5], dtype=np.int64))

  @pk.jit
  def bfactor_sums(r):
    out = np.empty(len(r), dtype=np.float32)
    for i, chain in enumerate(r):
      out[i] = np.sum(chain.bfactor)
    return out

  np.testing.assert_allclose(bfactor_sums(res), np.array([0.75, 1.0], dtype=np.float32))


def test_ragged_invalid_offsets_checked():
  @pk.jit
  def sum_rag(rag):
    return dsl.segmented_reduce("add", rag, init=0.0)

  # Non-monotonic offsets
  bad_mono = RaggedArray(np.ones(4, dtype=np.float32), np.array([0, 3, 2, 4], dtype=np.int64))
  with pytest.raises(ValueError, match="non-decreasing"):
    sum_rag(bad_mono)

  # offsets[-1] > len(values)
  bad_oob = RaggedArray(np.ones(4, dtype=np.float32), np.array([0, 2, 5], dtype=np.int64))
  with pytest.raises(ValueError, match="exceeds"):
    sum_rag(bad_oob)

  # offsets[0] != 0
  bad_start = RaggedArray(np.ones(4, dtype=np.float32), np.array([1, 2, 4], dtype=np.int64))
  with pytest.raises(ValueError, match="offsets\\[0\\]"):
    sum_rag(bad_start)


# ---------------------------------------------------------------------------
# 2. Segmented operators: map, reduce, scan, softmax, segment_ids
# ---------------------------------------------------------------------------


def test_segmented_operators():
  r = RaggedArray.from_sequences([[1.0, 2.0, 3.0], [], [4.0], [-2.0, 5.0, 1.0]], dtype=np.float32)

  @pk.jit
  def run_all(rag):
    mapped = dsl.segmented_map(lambda x: x * np.float32(2.0), rag)
    sums = dsl.segmented_reduce("add", mapped, init=np.float32(0.0))
    means = dsl.segmented_reduce("mean", rag, init=np.float32(0.0))
    inc = dsl.segmented_scan("add", rag)
    exc = dsl.segmented_scan("add", rag, exclusive=True)
    sm = dsl.segmented_softmax(rag)
    sids = dsl.segment_ids(rag)
    return mapped, sums, means, inc, exc, sm, sids

  mapped, sums, means, inc, exc, sm, sids = run_all(r)
  assert isinstance(mapped, RaggedArray)
  np.testing.assert_allclose(mapped.values, r.values * 2.0)
  np.testing.assert_allclose(sums, np.array([12.0, 0.0, 8.0, 8.0], dtype=np.float32))
  np.testing.assert_allclose(means, np.array([2.0, 0.0, 4.0, 4.0 / 3.0], dtype=np.float32), rtol=1e-6)
  np.testing.assert_allclose(inc.values, np.array([1.0, 3.0, 6.0, 4.0, -2.0, 3.0, 4.0], dtype=np.float32))
  np.testing.assert_allclose(exc.values, np.array([0.0, 1.0, 3.0, 0.0, 0.0, -2.0, 3.0], dtype=np.float32))
  np.testing.assert_array_equal(sids, np.array([0, 0, 0, 2, 3, 3, 3], dtype=np.int64))

  # Verify per-segment softmax
  for b in range(len(r)):
    seg = r[b]
    if len(seg) > 0:
      e = np.exp(seg - seg.max())
      np.testing.assert_allclose(sm[b], e / e.sum(), rtol=1e-6, atol=1e-6)

  # GNN-style scatter pooling using segment_ids
  @pk.jit
  def gnn_pool(rag):
    sids = dsl.segment_ids(rag)
    return pk.scatter_add(rag.num_segments, sids, rag.values)

  np.testing.assert_allclose(gnn_pool(r), np.array([6.0, 0.0, 4.0, 4.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# 3. Reverse-mode AD through RaggedArray & segmented operators
# ---------------------------------------------------------------------------


def _num_grad(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
  g = np.zeros_like(x)
  flat_x = x.ravel()
  flat_g = g.ravel()
  for i in range(flat_x.size):
    orig = flat_x[i]
    flat_x[i] = orig + eps
    fp = float(fn(x))
    flat_x[i] = orig - eps
    fm = float(fn(x))
    flat_x[i] = orig
    flat_g[i] = (fp - fm) / (2 * eps)
  return g


def test_autodiff_segmented_reduce_and_softmax():
  offsets = np.array([0, 3, 3, 5, 8], dtype=np.int64)
  vals = np.array([0.3, -1.2, 0.7, 2.1, -0.4, 0.5, 1.1, -0.8], dtype=np.float64)
  weights = np.array([1.0, -0.5, 2.0, 0.3, 1.5, -1.0, 0.8, 0.4], dtype=np.float64)
  r = RaggedArray(vals, offsets)

  def loss_reduce(rag):
    s = dsl.segmented_reduce("add", rag, init=0.0)
    m = dsl.segmented_reduce("mean", rag, init=0.0)
    return np.sum(s * s) + np.sum(m)

  def loss_softmax(rag):
    p = dsl.segmented_softmax(rag)
    return np.sum(p.values * weights)

  g_red = pk.grad(loss_reduce)(r)
  assert isinstance(g_red, RaggedArray)
  np.testing.assert_array_equal(g_red.offsets, offsets)
  ref_red = _num_grad(lambda v: pk.jit(loss_reduce)(RaggedArray(v.copy(), offsets)), vals.copy())
  np.testing.assert_allclose(g_red.values, ref_red, rtol=1e-5, atol=1e-5)

  g_sm = pk.grad(loss_softmax)(r)
  assert isinstance(g_sm, RaggedArray)
  ref_sm = _num_grad(lambda v: pk.jit(loss_softmax)(RaggedArray(v.copy(), offsets)), vals.copy())
  np.testing.assert_allclose(g_sm.values, ref_sm, rtol=1e-5, atol=1e-5)

  # Max reduction gradient
  offsets_ne = np.array([0, 3, 5, 8], dtype=np.int64)
  r_ne = RaggedArray(vals, offsets_ne)

  def loss_max(rag):
    return np.sum(dsl.segmented_reduce("max", rag))

  g_max = pk.grad(loss_max)(r_ne)
  ref_max = _num_grad(lambda v: pk.jit(loss_max)(RaggedArray(v.copy(), offsets_ne)), vals.copy())
  np.testing.assert_allclose(g_max.values, ref_max, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Varlen FlashAttention forward + Law 2 D_i-recomputed backward
# ---------------------------------------------------------------------------


def _ref_varlen_attn(q, k, v, cu_seqlens, *, scale=None, causal=False, window=None):
  total, H, D = q.shape
  out = np.zeros((total, H, v.shape[-1]), dtype=np.float64)
  sc = (1.0 / np.sqrt(D)) if scale is None else float(scale)
  for b in range(len(cu_seqlens) - 1):
    s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
    if s == e:
      continue
    L = e - s
    qb = q[s:e].astype(np.float64).transpose(1, 0, 2)  # (H, L, D)
    kb = k[s:e].astype(np.float64).transpose(1, 0, 2)
    vb = v[s:e].astype(np.float64).transpose(1, 0, 2)
    scores = np.matmul(qb, np.swapaxes(kb, -1, -2)) * sc
    idx = np.arange(L)
    mask = np.ones((L, L), dtype=bool)
    if causal:
      mask &= idx[:, None] >= idx[None, :]
    if window is not None:
      mask &= (idx[:, None] - idx[None, :]) <= window
      if not causal:
        mask &= (idx[None, :] - idx[:, None]) <= window
    scores = np.where(mask[None, :, :], scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out[s:e] = np.matmul(p, vb).transpose(1, 0, 2)
  return out.astype(q.dtype)


@pytest.mark.parametrize("causal,window", [(False, None), (True, None), (True, 3), (False, 2)])
def test_varlen_flash_attention_forward_and_backward(causal, window):
  rng = np.random.default_rng(42)
  lengths = [5, 0, 1, 7, 4]
  cu = np.zeros(len(lengths) + 1, dtype=np.int64)
  np.cumsum(lengths, out=cu[1:])
  total = int(cu[-1])
  H, K_dim, D_dim = 2, 6, 4
  q = rng.standard_normal((total, H, K_dim)).astype(np.float64) * 0.3
  k = rng.standard_normal((total, H, K_dim)).astype(np.float64) * 0.3
  v = rng.standard_normal((total, H, D_dim)).astype(np.float64) * 0.3
  w_loss = rng.standard_normal((total, H, D_dim)).astype(np.float64)

  @pk.jit
  def fwd(q_, k_, v_, cu_):
    return dsl.varlen_flash_attention(q_, k_, v_, cu_, causal=causal, window=window)

  out = fwd(q, k, v, cu)
  ref = _ref_varlen_attn(q, k, v, cu, causal=causal, window=window)
  np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

  # RaggedArray wrapper form + Law 2 backward via pk.grad
  def loss_fn(q_, k_, v_):
    o = dsl.varlen_flash_attention(q_, k_, v_, cu, causal=causal, window=window)
    return np.sum(o * w_loss)

  dq, dk, dv = pk.grad(loss_fn, argnum=(0, 1, 2))(q, k, v)
  ref_dq = _num_grad(
    lambda x: float(np.sum(_ref_varlen_attn(x, k, v, cu, causal=causal, window=window) * w_loss)), q.copy()
  )
  ref_dk = _num_grad(
    lambda x: float(np.sum(_ref_varlen_attn(q, x, v, cu, causal=causal, window=window) * w_loss)), k.copy()
  )
  ref_dv = _num_grad(
    lambda x: float(np.sum(_ref_varlen_attn(q, k, x, cu, causal=causal, window=window) * w_loss)), v.copy()
  )
  np.testing.assert_allclose(dq, ref_dq, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dk, ref_dk, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dv, ref_dv, rtol=1e-4, atol=1e-4)

  # RaggedArray direct call + grad w.r.t. RaggedArray
  rq = RaggedArray(q, cu)
  rk = RaggedArray(k, cu)
  rv = RaggedArray(v, cu)
  ro = pk.jit(lambda a, b, c: dsl.varlen_flash_attention(a, b, c, causal=causal, window=window))(rq, rk, rv)
  assert isinstance(ro, RaggedArray)
  np.testing.assert_allclose(ro.values, ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. GPU verification (to_device/to_host, segmented ops, varlen FlashAttention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_ragged_and_varlen_attention_on_gpu(backend):
  if backend == "cuda" and not os.environ.get("PARAKEET_CUDA_SIMULATE"):
    if pk.available_backends().get("cuda") != "ok":
      pytest.skip("CUDA not available")
  if backend == "metal" and pk.available_backends().get("metal") != "ok":
    pytest.skip("Metal not available")

  rng = np.random.default_rng(11)
  lengths = [12, 0, 7, 19, 5]
  cu = np.zeros(len(lengths) + 1, dtype=np.int64)
  np.cumsum(lengths, out=cu[1:])
  total = int(cu[-1])
  vals = rng.standard_normal(total).astype(np.float32)
  rag = RaggedArray(vals, cu)

  q = rng.standard_normal((total, 4, 16)).astype(np.float32) * np.float32(0.25)
  k = rng.standard_normal((total, 4, 16)).astype(np.float32) * np.float32(0.25)
  v = rng.standard_normal((total, 4, 16)).astype(np.float32) * np.float32(0.25)

  @pk.jit(backend=backend)
  def gpu_seg_ops(r):
    sm = dsl.segmented_softmax(r)
    red = dsl.segmented_reduce("add", sm, init=np.float32(0.0))
    return sm, red

  @pk.jit(backend=backend)
  def gpu_vfa(q_, k_, v_, cu_):
    return dsl.varlen_flash_attention(q_, k_, v_, cu_, causal=True)

  rag_dev = pk.to_device(rag, backend=backend)
  with pk.options(device_fallback="error", device_threshold=1):
    sm_dev, red_dev = gpu_seg_ops(rag_dev)
    sm_h, red_h = pk.to_host((sm_dev, red_dev))
    out_dev = gpu_vfa(*pk.to_device((q, k, v, cu), backend=backend))
    out_h = pk.to_host(out_dev)

  assert isinstance(sm_h, RaggedArray)
  np.testing.assert_allclose(red_h, np.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32), rtol=1e-5, atol=1e-5)
  ref = _ref_varlen_attn(q, k, v, cu, causal=True)
  np.testing.assert_allclose(out_h, ref, rtol=1e-4, atol=1e-4)
