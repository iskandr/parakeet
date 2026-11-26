"""Tests for compiled np.linalg builtins (cholesky, det, solve, inv, qr, eigh, svd)
and the classic bioinformatics algorithms suite in examples/bio_algorithms.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import parakeet as par
from examples.bio_algorithms import (
  batched_kabsch_rmsd,
  build_fm_index,
  expand_cigar,
  fm_backward_search_batch,
  gotoh_global_align,
  hmm_viterbi_and_posterior,
  kabsch_rmsd,
  minimizer_sketch,
  myers_edit_distance,
)


# ---------------------------------------------------------------------------
# 1. Compiled np.linalg builtins (cholesky, det, solve, inv, qr, eigh, svd)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linalg_cholesky_solve_inv_det_qr(dtype):
  @par.jit
  def run_all(A_spd, A_gen, b1, b2):
    L = np.linalg.cholesky(A_spd)
    d = np.linalg.det(A_gen)
    x1 = np.linalg.solve(A_gen, b1)
    x2 = np.linalg.solve(A_gen, b2)
    Ainv = np.linalg.inv(A_gen)
    Q, R = np.linalg.qr(A_gen)
    return L, d, x1, x2, Ainv, Q, R

  rng = np.random.default_rng(7)
  M = rng.standard_normal((6, 6)).astype(dtype)
  A_spd = (M.T @ M + dtype(2.0) * np.eye(6, dtype=dtype)).astype(dtype)
  A_gen = (rng.standard_normal((6, 6)) + 3.0 * np.eye(6)).astype(dtype)
  b1 = rng.standard_normal(6).astype(dtype)
  b2 = rng.standard_normal((6, 3)).astype(dtype)

  tol = 1e-4 if dtype == np.float32 else 1e-11
  L, d, x1, x2, Ainv, Q, R = run_all(A_spd, A_gen, b1, b2)
  np.testing.assert_allclose(L, np.linalg.cholesky(A_spd), rtol=tol, atol=tol)
  np.testing.assert_allclose(d, np.linalg.det(A_gen), rtol=tol, atol=tol)
  np.testing.assert_allclose(x1, np.linalg.solve(A_gen, b1), rtol=tol, atol=tol)
  np.testing.assert_allclose(x2, np.linalg.solve(A_gen, b2), rtol=tol, atol=tol)
  np.testing.assert_allclose(Ainv, np.linalg.inv(A_gen), rtol=tol, atol=tol)
  np.testing.assert_allclose(Q @ R, A_gen, rtol=tol, atol=tol)
  np.testing.assert_allclose(Q.T @ Q, np.eye(6, dtype=dtype), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(3, 3), (6, 4), (4, 6), (5, 5)])
def test_linalg_eigh_and_svd(dtype, shape):
  @par.jit
  def do_eigh(A):
    return np.linalg.eigh(A)

  @par.jit
  def do_svd(A):
    return np.linalg.svd(A, full_matrices=True)

  @par.jit
  def do_svd_reduced(A):
    return np.linalg.svd(A, full_matrices=False)

  @par.jit
  def do_svd_vals(A):
    return np.linalg.svd(A, compute_uv=False)

  rng = np.random.default_rng(13 + shape[0] * 7 + shape[1])
  tol = 1e-4 if dtype == np.float32 else 1e-11

  A = rng.standard_normal(shape).astype(dtype)
  U, S, Vt = do_svd(A)
  U_r, S_r, Vt_r = do_svd_reduced(A)
  S_only = do_svd_vals(A)
  ref_S = np.linalg.svd(A, compute_uv=False)

  m, n = shape
  k = min(m, n)
  assert U.shape == (m, m) and S.shape == (k,) and Vt.shape == (n, n)
  assert U_r.shape == (m, k) and Vt_r.shape == (k, n)
  np.testing.assert_allclose(S, ref_S, rtol=tol, atol=tol)
  np.testing.assert_allclose(S_r, ref_S, rtol=tol, atol=tol)
  np.testing.assert_allclose(S_only, ref_S, rtol=tol, atol=tol)
  np.testing.assert_allclose(U_r @ np.diag(S_r) @ Vt_r, A, rtol=tol, atol=tol)
  np.testing.assert_allclose(U.T @ U, np.eye(m, dtype=dtype), rtol=tol, atol=tol)
  np.testing.assert_allclose(Vt @ Vt.T, np.eye(n, dtype=dtype), rtol=tol, atol=tol)

  if m == n:
    sym = (0.5 * (A + A.T)).astype(dtype)
    w, V = do_eigh(sym)
    ref_w, _ = np.linalg.eigh(sym)
    np.testing.assert_allclose(w, ref_w, rtol=tol, atol=tol)
    np.testing.assert_allclose(V.T @ V, np.eye(n, dtype=dtype), rtol=tol, atol=tol)
    np.testing.assert_allclose(V @ np.diag(w) @ V.T, sym, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# 2. Classic bioinformatics algorithms verification
# ---------------------------------------------------------------------------


def _ref_levenshtein(a: bytes, b: bytes) -> int:
  n, m = len(a), len(b)
  dp = list(range(m + 1))
  for i in range(1, n + 1):
    prev = dp[0]
    dp[0] = i
    ai = a[i - 1]
    for j in range(1, m + 1):
      cur = dp[j]
      cost = 0 if ai == b[j - 1] else 1
      dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
      prev = cur
  return dp[m]


def test_gotoh_alignment_traceback_and_cigar_expand():
  a = np.frombuffer(b"ACGTACGTAAAACGT", dtype=np.uint8)
  b = np.frombuffer(b"ACGTCGTAAAAGTCGT", dtype=np.uint8)
  score, ops, lens = gotoh_global_align(a, b, 2, -1, -4, -1)
  ref_score, ref_ops, ref_lens = gotoh_global_align(a, b, 2, -1, -4, -1, _backend="python")
  assert int(score) == int(ref_score)
  np.testing.assert_array_equal(ops, ref_ops)
  np.testing.assert_array_equal(lens, ref_lens)

  # Reconstruct alignment from expanded CIGAR and verify score matches DP score exactly!
  flat = expand_cigar(ops, lens)
  ia = ib = 0
  recomputed = 0
  prev_op = -1
  for op in flat:
    op = int(op)
    if op == 0:
      recomputed += 2 if a[ia] == b[ib] else -1
      ia += 1
      ib += 1
    elif op == 1:
      recomputed += -1 if prev_op == 1 else -5
      ia += 1
    else:
      recomputed += -1 if prev_op == 2 else -5
      ib += 1
    prev_op = op
  assert ia == len(a) and ib == len(b)
  assert recomputed == int(score)


def test_myers_bitvector_single_and_multi_word():
  rng = np.random.default_rng(19)
  alph = np.frombuffer(b"ACGT", dtype=np.uint8)
  for pat_len in (1, 15, 64, 65, 130):
    pat = alph[rng.integers(0, 4, size=pat_len)]
    txt = alph[rng.integers(0, 4, size=pat_len + 17)]
    got = int(myers_edit_distance(pat, txt))
    ref = _ref_levenshtein(bytes(pat), bytes(txt))
    assert got == ref, (pat_len, got, ref)


def test_minimizer_sketch_exact_vs_naive():
  rng = np.random.default_rng(23)
  alph = np.frombuffer(b"ACGT", dtype=np.uint8)
  seq = alph[rng.integers(0, 4, size=300)]
  # Insert an 'N' break in the middle
  seq[120] = ord("N")
  h_jit, p_jit, s_jit = minimizer_sketch(seq, 11, 8)
  h_py, p_py, s_py = minimizer_sketch(seq, 11, 8, _backend="python")
  np.testing.assert_array_equal(h_jit, h_py)
  np.testing.assert_array_equal(p_jit, p_py)
  np.testing.assert_array_equal(s_jit, s_py)


def test_fm_index_backward_search_exact():
  rng = np.random.default_rng(31)
  text = rng.integers(1, 5, size=250, dtype=np.uint8)
  text_with_term = np.concatenate([text, np.array([0], dtype=np.uint8)])
  bwt, C, occ, sa, stride = build_fm_index(text_with_term, alphabet_size=5, occ_stride=16)

  raw_queries = [text[10:15], text[40:44], text[100:106], np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)]
  rag = par.RaggedArray.from_sequences(raw_queries)
  lo, hi = fm_backward_search_batch(bwt, C, occ, stride, rag)

  t_bytes = text.tobytes()
  for idx, q in enumerate(raw_queries):
    qb = q.tobytes()
    expected = sum(1 for i in range(len(t_bytes) - len(qb) + 1) if t_bytes[i : i + len(qb)] == qb)
    assert int(hi[idx] - lo[idx]) == expected
    for pos in sa[int(lo[idx]) : int(hi[idx])]:
      assert text[pos : pos + len(q)].tobytes() == qb


def test_profile_hmm_viterbi_and_posterior():
  rng = np.random.default_rng(3)
  S, V, T = 4, 6, 30
  init = rng.dirichlet(np.ones(S))
  trans = rng.dirichlet(np.ones(S), size=S)
  emit = rng.dirichlet(np.ones(V), size=S)
  obs = rng.integers(0, V, size=T, dtype=np.int64)

  path, vlp, tlp, post = hmm_viterbi_and_posterior(obs, np.log(init), np.log(trans), np.log(emit))
  ref_path, ref_vlp, ref_tlp, ref_post = hmm_viterbi_and_posterior(
    obs, np.log(init), np.log(trans), np.log(emit), _backend="python"
  )
  np.testing.assert_array_equal(path, ref_path)
  np.testing.assert_allclose(vlp, ref_vlp, rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(tlp, ref_tlp, rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(post, ref_post, rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(post.sum(axis=1), np.ones(T), rtol=1e-11)


def test_kabsch_3d_rmsd_and_reflection_handling():
  rng = np.random.default_rng(99)
  Q = rng.standard_normal((64, 3))
  # Known proper rotation + translation + tiny noise
  A = rng.standard_normal((3, 3))
  R_true, _ = np.linalg.qr(A)
  if np.linalg.det(R_true) < 0:
    R_true[:, 2] *= -1.0
  P_exact = Q @ R_true.T + np.array([10.0, -5.0, 2.5])
  rmsd, R_est, P_al = kabsch_rmsd(P_exact, Q)
  assert float(rmsd) < 1e-12
  assert abs(float(np.linalg.det(R_est)) - 1.0) < 1e-12
  np.testing.assert_allclose(P_al, Q, atol=1e-11)

  # Mirror reflection must NEVER produce det(R) == -1 (proper rotation required!)
  P_mirror = Q.copy()
  P_mirror[:, 2] *= -1.0
  rmsd_m, R_m, _ = kabsch_rmsd(P_mirror, Q)
  assert abs(float(np.linalg.det(R_m)) - 1.0) < 1e-12
  assert float(rmsd_m) > 0.1

  # Batched parallel Kabsch RMSD
  batch = np.stack([P_exact, P_mirror, Q + 0.05 * rng.standard_normal(Q.shape)], axis=0)
  rmsds = batched_kabsch_rmsd(batch, Q, _backend="multicore")
  assert float(rmsds[0]) < 1e-12
  np.testing.assert_allclose(float(rmsds[1]), float(rmsd_m), rtol=1e-11)
  assert 0.0 < float(rmsds[2]) < 0.2


def test_patro_sshash_longread_rna_pipeline():
  from examples.patro_longread_rna import (
    build_unitig_sshash_index,
    make_synthetic_isoforms,
    oarfish_em_quantify,
    simulate_long_reads,
    stream_map_long_reads,
  )

  rng = np.random.default_rng(11)
  _, seqs = make_synthetic_isoforms(rng, n_genes=4, isoforms_per_gene=4, exon_len=180)
  idx = build_unitig_sshash_index(seqs, k=15, m=9)
  assert idx["unitig_bp"] < idx["raw_bp"]

  reads_rag, true_tx, true_counts = simulate_long_reads(seqs, n_reads=300, sub_rate=0.015, indel_rate=0.005, seed=3)
  read_lens = np.diff(reads_rag.offsets).astype(np.float64)

  best_tx, best_sc, compat_c, n_ext, n_probe, tx_scores = stream_map_long_reads(
    reads_rag,
    idx["u_vals"],
    idx["u_offs"],
    idx["p_tx"],
    idx["p_pos"],
    idx["p_offs"],
    idx["min_table"],
    idx["b_offs"],
    idx["b_uid"],
    idx["b_mpos"],
    idx["n_tx"],
    _backend="multicore",
  )
  est = oarfish_em_quantify(tx_scores, read_lens, idx["tx_lens"], _backend="multicore")

  mapped = float(np.mean(best_tx >= 0))
  true_in_delta = float(np.mean([tx_scores[i, true_tx[i]] > 0 for i in range(len(true_tx))]))
  skip_rate = float(n_ext) / max(1.0, float(n_ext + n_probe))
  corr = float(np.corrcoef(true_counts, est)[0, 1])

  assert mapped > 0.98
  assert true_in_delta > 0.95
  assert skip_rate > 0.85
  assert corr > 0.95
