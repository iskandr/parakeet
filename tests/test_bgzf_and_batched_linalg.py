"""Tests for compiled parallel BGZF DEFLATE inflate, overlapped streaming FASTQ I/O,
batched 3D np.linalg.* builtins (cholesky, det, solve, inv, qr, eigh, eigvalsh, svd,
pinv, lstsq), and differentiable protein backbone dihedral angles.
"""

from __future__ import annotations

import gzip

import numpy as np
import pytest

import parakeet as par
from examples.bgzf_and_streaming_io import (
  fastq_slab_qc,
  index_bgzf_blocks,
  inflate_bgzf_blocks,
  make_bgzf_bytes,
  make_synthetic_fastq,
  stream_fastq_overlapped,
)
from examples.bio_algorithms import backbone_dihedrals


# ---------------------------------------------------------------------------
# 1. Compiled RFC-1951 parallel BGZF inflate (stored, fixed, dynamic, LZ77 RLE)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 6, 9])
@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_bgzf_inflate_all_deflate_block_types(level, backend):
  # Mix repetitive RLE (dist < len), short fixed-Huffman chunks, and real FASTQ text
  chunks = [
    b"A" * 50_000 + b"CGT" * 10_000,  # exercises overlapping LZ77 distance < length
    b"tiny fixed huffman block\n",
    make_synthetic_fastq(n_reads=600, read_len=120, seed=3),
  ]
  for raw in chunks:
    bgzf = make_bgzf_bytes(raw, block_size=16384, level=level)
    assert gzip.decompress(bgzf) == raw
    data, cs, ce, uo = index_bgzf_blocks(bgzf)
    got = inflate_bgzf_blocks(data, cs, ce, uo, _backend=backend)
    assert got.tobytes() == raw


def test_overlapped_fastq_stream_exact_across_slab_boundaries():
  raw_fastq = make_synthetic_fastq(n_reads=800, read_len=100, seed=5)
  bgzf = make_bgzf_bytes(raw_fastq, block_size=8192, level=6)
  ref = tuple(int(x) for x in fastq_slab_qc(np.frombuffer(raw_fastq, dtype=np.uint8), _backend="multicore"))
  # Test slab sizes that split mid-FASTQ-record to exercise the 4-line carry logic
  for blocks_per_slab in (1, 3, 16):
    got = stream_fastq_overlapped(bgzf, blocks_per_slab=blocks_per_slab)
    assert got == ref, (blocks_per_slab, got, ref)


# ---------------------------------------------------------------------------
# 2. Batched 3D np.linalg.* + eigvalsh, pinv, lstsq
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_3d_linalg_and_pinv_lstsq_eigvalsh(dtype):
  @par.jit
  def run_batched(A_spd, A_gen, b_vec, A_tall, A_wide, b_ls):
    L = np.linalg.cholesky(A_spd)
    d = np.linalg.det(A_gen)
    x = np.linalg.solve(A_gen, b_vec)
    invA = np.linalg.inv(A_gen)
    Q, R = np.linalg.qr(A_gen)
    w, V = np.linalg.eigh(A_spd)
    w_only = np.linalg.eigvalsh(A_spd)
    U_t, S_t, Vt_t = np.linalg.svd(A_tall, full_matrices=False)
    U_w, S_w, Vt_w = np.linalg.svd(A_wide, full_matrices=False)
    P_t = np.linalg.pinv(A_tall)
    xls, res, rank, s_ls = np.linalg.lstsq(A_tall[0], b_ls)
    return L, d, x, invA, Q, R, w, V, w_only, U_t, S_t, Vt_t, U_w, S_w, Vt_w, P_t, xls, res, rank, s_ls

  rng = np.random.default_rng(19)
  B = 10
  M = rng.standard_normal((B, 4, 4)).astype(dtype)
  A_spd = (np.matmul(M.swapaxes(1, 2), M) + dtype(2.0) * np.eye(4, dtype=dtype)[None]).astype(dtype)
  A_gen = (rng.standard_normal((B, 4, 4)) + 3.0 * np.eye(4)[None]).astype(dtype)
  b_vec = rng.standard_normal((B, 4)).astype(dtype)
  A_tall = rng.standard_normal((B, 5, 3)).astype(dtype)
  A_wide = rng.standard_normal((B, 3, 5)).astype(dtype)
  b_ls = rng.standard_normal(5).astype(dtype)

  tol = 2e-4 if dtype == np.float32 else 1e-10
  out = run_batched(A_spd, A_gen, b_vec, A_tall, A_wide, b_ls)
  L, d, x, invA, Q, R, w, V, w_only, U_t, S_t, Vt_t, U_w, S_w, Vt_w, P_t, xls, res, rank, s_ls = out

  np.testing.assert_allclose(L, np.linalg.cholesky(A_spd), rtol=tol, atol=tol)
  np.testing.assert_allclose(d, np.linalg.det(A_gen), rtol=tol, atol=tol)
  np.testing.assert_allclose(x, np.linalg.solve(A_gen, b_vec[..., None])[..., 0], rtol=tol, atol=tol)
  np.testing.assert_allclose(invA, np.linalg.inv(A_gen), rtol=tol, atol=tol)
  np.testing.assert_allclose(np.matmul(Q, R), A_gen, rtol=tol, atol=tol)
  np.testing.assert_allclose(w, np.linalg.eigvalsh(A_spd), rtol=tol, atol=tol)
  np.testing.assert_allclose(w_only, np.linalg.eigvalsh(A_spd), rtol=tol, atol=tol)
  np.testing.assert_allclose(S_t, np.linalg.svd(A_tall, compute_uv=False), rtol=tol, atol=tol)
  np.testing.assert_allclose(S_w, np.linalg.svd(A_wide, compute_uv=False), rtol=tol, atol=tol)
  np.testing.assert_allclose(P_t, np.linalg.pinv(A_tall), rtol=tol, atol=tol)
  ref_xls, ref_res, ref_rank, _ = np.linalg.lstsq(A_tall[0], b_ls, rcond=None)
  np.testing.assert_allclose(xls, ref_xls, rtol=tol, atol=tol)
  np.testing.assert_allclose(res, ref_res, rtol=tol, atol=tol)
  assert int(rank) == int(ref_rank)


# ---------------------------------------------------------------------------
# 3. Protein backbone Ramachandran dihedral angles + par.grad verification
# ---------------------------------------------------------------------------


def test_backbone_dihedrals_exact_and_differentiable():
  rng = np.random.default_rng(7)
  L = 24
  N = rng.standard_normal((L, 3))
  CA = N + np.array([1.46, 0.0, 0.0]) + 0.3 * rng.standard_normal((L, 3))
  C = CA + np.array([0.5, 1.4, 0.2]) + 0.3 * rng.standard_normal((L, 3))

  phi, psi, omega = backbone_dihedrals(N, CA, C)
  ref_phi, ref_psi, ref_omega = backbone_dihedrals(N, CA, C, _backend="python")
  np.testing.assert_allclose(phi, ref_phi, rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(psi, ref_psi, rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(omega, ref_omega, rtol=1e-11, atol=1e-11)

  # Reverse-mode gradient through torsion angles vs finite differences
  def torsion_energy(N_, CA_, C_):
    p, s, o = backbone_dihedrals(N_, CA_, C_)
    return np.sum(np.cos(p) + np.sin(s) + np.cos(o))

  g_CA = par.grad(torsion_energy, argnum=1)(N, CA, C)
  eps = 1e-6
  CA_p, CA_m = CA.copy(), CA.copy()
  CA_p[5, 1] += eps
  CA_m[5, 1] -= eps
  fd = (float(torsion_energy(N, CA_p, C)) - float(torsion_energy(N, CA_m, C))) / (2 * eps)
  np.testing.assert_allclose(g_CA[5, 1], fd, rtol=1e-5, atol=1e-6)
