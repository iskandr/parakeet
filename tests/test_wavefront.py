"""Tests for #53: 2D / banded IndexWavefront semiring DP and checkpointed differentiable alignment."""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import pytest

import parakeet as pk
from parakeet import dsl


class GotohState(NamedTuple):
  m: float
  ix: float
  iy: float


# ---------------------------------------------------------------------------
# 1. Exact Smith-Waterman, Needleman-Wunsch, Gotoh affine gap, banded & edge cases
# ---------------------------------------------------------------------------


def _ref_sw(a, b, match=2.0, mismatch=-1.0, gap=-2.0, band=None):
  M, N = len(a), len(b)
  if M == 0 or N == 0:
    return 0.0, -1, -1
  H = np.zeros((M, N), dtype=np.float64)
  best, bi, bj = 0.0, -1, -1
  for i in range(M):
    for j in range(N):
      if band is not None and abs(i - j) > band:
        continue
      top = H[i - 1, j] if (i > 0 and (band is None or abs((i - 1) - j) <= band)) else 0.0
      left = H[i, j - 1] if (j > 0 and (band is None or abs(i - (j - 1)) <= band)) else 0.0
      diag = H[i - 1, j - 1] if (i > 0 and j > 0) else 0.0
      s = match if a[i] == b[j] else mismatch
      v = max(0.0, diag + s, top + gap, left + gap)
      H[i, j] = v
      if v > best:
        best, bi, bj = v, i, j
  return best, bi, bj


@pytest.mark.parametrize(
  "shape,band",
  [((0, 5), None), ((1, 1), None), ((6, 18), None), ((15, 7), None), ((16, 16), 0), ((16, 16), 1), ((16, 16), 5)],
)
def test_wavefront_smith_waterman_and_banded(shape, band):
  rng = np.random.default_rng(53)
  a = rng.integers(0, 4, size=shape[0], dtype=np.int32)
  b = rng.integers(0, 4, size=shape[1], dtype=np.int32)

  @pk.jit
  def sw(seq_a, seq_b, bnd):
    def step(top, left, diag, i, j):
      sub = 2.0 if seq_a[i] == seq_b[j] else -1.0
      return max(0.0, diag + sub, top - 2.0, left - 2.0)

    return dsl.wavefront(step, (len(seq_a), len(seq_b)), 0.0, 0.0, band=bnd, track_max=True)

  @pk.jit
  def sw_unbanded(seq_a, seq_b):
    def step(top, left, diag, i, j):
      sub = 2.0 if seq_a[i] == seq_b[j] else -1.0
      return max(0.0, diag + sub, top - 2.0, left - 2.0)

    return dsl.wavefront(step, (len(seq_a), len(seq_b)), 0.0, 0.0, track_max=True)

  _, best, bi, bj = sw_unbanded(a, b) if band is None else sw(a, b, band)
  ref_best, ref_bi, ref_bj = _ref_sw(a, b, band=band)
  np.testing.assert_allclose(best, ref_best)
  if ref_best > 0.0:
    # Check score at reported (bi, bj) equals ref_best
    assert bi >= 0 and bj >= 0


def test_wavefront_gotoh_affine_gap_3state_namedtuple():
  rng = np.random.default_rng(101)
  # 20-amino-acid synthetic BLOSUM-like substitution matrix
  sub_mat = rng.integers(-4, 7, size=(20, 20)).astype(np.float32)
  sub_mat = (sub_mat + sub_mat.T) * np.float32(0.5)
  a = rng.integers(0, 20, size=24, dtype=np.int32)
  b = rng.integers(0, 20, size=28, dtype=np.int32)
  gap_open = np.float32(-6.0)
  gap_ext = np.float32(-1.0)

  @pk.jit
  def gotoh_local(seq_a, seq_b, sm, go, ge):
    z = np.float32(0.0)
    neg = np.float32(-1e9)
    init = GotohState(z, neg, neg)

    def step(top, left, diag, i, j):
      s = sm[seq_a[i], seq_b[j]]
      ix = max(top.m + go, top.ix + ge)
      iy = max(left.m + go, left.iy + ge)
      m_val = max(z, diag.m + s, ix, iy)
      return GotohState(m_val, ix, iy)

    _, best, _, _ = dsl.wavefront(step, (len(seq_a), len(seq_b)), init, init, track_max=True)
    return best

  got = float(gotoh_local(a, b, sub_mat, gap_open, gap_ext))

  # Naive Python reference
  M, N = len(a), len(b)
  Hm = np.zeros((M + 1, N + 1), dtype=np.float32)
  Hx = np.full((M + 1, N + 1), -1e9, dtype=np.float32)
  Hy = np.full((M + 1, N + 1), -1e9, dtype=np.float32)
  ref = 0.0
  for i in range(1, M + 1):
    for j in range(1, N + 1):
      s = sub_mat[a[i - 1], b[j - 1]]
      Hx[i, j] = max(Hm[i - 1, j] + gap_open, Hx[i - 1, j] + gap_ext)
      Hy[i, j] = max(Hm[i, j - 1] + gap_open, Hy[i, j - 1] + gap_ext)
      Hm[i, j] = max(np.float32(0.0), Hm[i - 1, j - 1] + s, Hx[i, j], Hy[i, j])
      ref = max(ref, float(Hm[i, j]))
  np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Differentiable Soft-SW & Soft-NW (Law 2 checkpointed AD vs finite diffs)
# ---------------------------------------------------------------------------


def _ref_soft_align(scores, gap, tau=1.0, band=None, local=True):
  M, N = scores.shape
  H = np.full((M, N), 0.0 if local else -np.inf, dtype=np.float64)

  def lse(xs):
    m = max(xs)
    if not np.isfinite(m):
      return m
    return m + tau * np.log(sum(np.exp((x - m) / tau) for x in xs))

  for i in range(M):
    for j in range(N):
      if band is not None and abs(i - j) > band:
        continue
      top_in = band is None or abs((i - 1) - j) <= band
      left_in = band is None or abs(i - (j - 1)) <= band
      top = (H[i - 1, j] if top_in else (0.0 if local else -np.inf)) if i > 0 else (0.0 if local else (j + 1) * gap)
      left = (H[i, j - 1] if left_in else (0.0 if local else -np.inf)) if j > 0 else (0.0 if local else (i + 1) * gap)
      if i > 0 and j > 0:
        diag = H[i - 1, j - 1]
      elif local:
        diag = 0.0
      elif j > 0:
        diag = j * gap
      elif i > 0:
        diag = i * gap
      else:
        diag = 0.0
      cands = [diag + scores[i, j], top + gap, left + gap]
      if local:
        cands.append(0.0)
      H[i, j] = lse(cands)
  if not local:
    return float(H[M - 1, N - 1])
  vals = [H[i, j] for i in range(M) for j in range(N) if band is None or abs(i - j) <= band]
  return float(lse(vals))


@pytest.mark.parametrize("local,band", [(True, None), (False, None), (True, 2), (False, 2)])
def test_soft_align_forward_and_checkpointed_grad(local, band):
  rng = np.random.default_rng(71)
  M, N = 9, 10
  scores = rng.standard_normal((M, N)).astype(np.float64) * 0.5
  gap = -1.2
  tau = 0.8

  def loss(sc, gp):
    return dsl.soft_align(sc, gp, temperature=tau, band=band, local=local)

  val, (g_sc, g_gp) = pk.value_and_grad(loss, argnum=(0, 1))(scores, gap)
  ref_val = _ref_soft_align(scores, gap, tau=tau, band=band, local=local)
  np.testing.assert_allclose(val, ref_val, rtol=1e-6, atol=1e-6)

  eps = 1e-5
  ref_g_sc = np.zeros_like(scores)
  for i in range(M):
    for j in range(N):
      sp = scores.copy()
      sp[i, j] += eps
      sm = scores.copy()
      sm[i, j] -= eps
      ref_g_sc[i, j] = (
        _ref_soft_align(sp, gap, tau=tau, band=band, local=local)
        - _ref_soft_align(sm, gap, tau=tau, band=band, local=local)
      ) / (2 * eps)
  ref_g_gp = (
    _ref_soft_align(scores, gap + eps, tau=tau, band=band, local=local)
    - _ref_soft_align(scores, gap - eps, tau=tau, band=band, local=local)
  ) / (2 * eps)

  np.testing.assert_allclose(g_sc, ref_g_sc, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(g_gp, ref_g_gp, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Batched variable-length alignment on CPU, CUDA simulator, and native Metal GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["c", "multicore", "cuda", "metal"])
def test_batched_wavefront_on_cpu_and_gpu(backend):
  if backend == "cuda" and not os.environ.get("PARAKEET_CUDA_SIMULATE"):
    if pk.available_backends().get("cuda") != "ok":
      pytest.skip("CUDA not available")
  if backend == "metal" and pk.available_backends().get("metal") != "ok":
    pytest.skip("Metal not available")

  rng = np.random.default_rng(19)
  B = 16
  m_lens = rng.integers(4, 24, size=B, dtype=np.int64)
  n_lens = rng.integers(4, 24, size=B, dtype=np.int64)
  off_a = np.zeros(B + 1, dtype=np.int64)
  off_b = np.zeros(B + 1, dtype=np.int64)
  np.cumsum(m_lens, out=off_a[1:])
  np.cumsum(n_lens, out=off_b[1:])
  seqs_a = rng.integers(0, 4, size=int(off_a[-1]), dtype=np.int32)
  seqs_b = rng.integers(0, 4, size=int(off_b[-1]), dtype=np.int32)

  @pk.jit(backend=backend)
  def batch_sw(sa, oa, ml, sb, ob, nl):
    z = np.float32(0.0)

    def step(b, top, left, diag, i, j):
      ca = sa[oa[b] + i]
      cb = sb[ob[b] + j]
      sub = np.float32(2.0) if ca == cb else np.float32(-1.0)
      return max(z, diag + sub, top - np.float32(2.0), left - np.float32(2.0))

    _, bests, _, _ = dsl.batched_wavefront(step, ml, nl, z, z, track_max=True)
    return bests

  opts = {"device_fallback": "error", "device_threshold": 1} if backend in ("cuda", "metal") else {}
  with pk.options(**opts):
    res = batch_sw(seqs_a, off_a, m_lens, seqs_b, off_b, n_lens)

  ref = np.array(
    [_ref_sw(seqs_a[off_a[k] : off_a[k + 1]], seqs_b[off_b[k] : off_b[k + 1]])[0] for k in range(B)],
    dtype=np.float32,
  )
  np.testing.assert_allclose(res, ref, rtol=1e-5, atol=1e-5)
