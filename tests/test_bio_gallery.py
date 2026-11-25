import numpy as np
import pytest

import parakeet as pk
from examples.bio_gallery import (
  banded_smith_waterman,
  dna_ssm_scan,
  encode_canonical_kmers_and_gc,
  protein_coord_grad,
  protein_rbf_loss,
  protein_rbf_message_passing,
  single_pass_pileup,
  tanimoto_nearest_hit,
)
from parakeet import dsl


def _ref_canonical_kmers(seq_bytes: np.ndarray, qual_bytes: np.ndarray, k: int):
  code_map = {ord("A"): 0, ord("C"): 1, ord("G"): 2, ord("T"): 3}
  n_reads, read_len = seq_bytes.shape
  n_kmers = read_len - k + 1
  kmers = np.zeros((n_reads, n_kmers), dtype=np.uint64)
  gc = np.zeros((n_reads,), dtype=np.float32)
  mq = np.zeros((n_reads,), dtype=np.float32)
  mask = (1 << (2 * k)) - 1
  for r in range(n_reads):
    gc_c = 0
    fwd = rev = 0
    for pos in range(read_len):
      ch = int(seq_bytes[r, pos])
      c = code_map[ch]
      if ch in (ord("C"), ord("G")):
        gc_c += 1
      fwd = ((fwd << 2) | c) & mask
      rev = (rev >> 2) | ((3 ^ c) << (2 * (k - 1)))
      if pos >= k - 1:
        kmers[r, pos - k + 1] = min(fwd, rev)
    gc[r] = np.float32(gc_c) / np.float32(read_len)
    mq[r] = np.float32(np.sum(qual_bytes[r].astype(np.int64) - 33)) / np.float32(read_len)
  return kmers, gc, mq


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_bio_gallery_fastq_and_pileup(backend):
  rng = np.random.default_rng(11)
  alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
  reads = alphabet[rng.integers(0, 4, size=(12, 30))]
  quals = rng.integers(33 + 15, 33 + 41, size=(12, 30), dtype=np.uint8)

  km, gc, mq = encode_canonical_kmers_and_gc(reads, quals, 11, _backend=backend)
  ekm, egc, emq = _ref_canonical_kmers(reads, quals, 11)
  np.testing.assert_array_equal(km, ekm)
  np.testing.assert_allclose(gc, egc, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(mq, emq, rtol=1e-6, atol=1e-6)

  # Single-pass 6-channel pileup: verifies IndexScatter 6 -> 1 fusion!
  contig = 128
  pos = rng.integers(0, contig, size=(2000,), dtype=np.int64)
  bases = rng.integers(0, 4, size=(2000,), dtype=np.uint8)
  phred = rng.integers(20, 42, size=(2000,), dtype=np.uint8)
  rev = rng.integers(0, 2, size=(2000,), dtype=np.uint8)

  rep = dsl.show_rewrite(single_pass_pileup, pos, bases, phred, rev, contig, backend=backend)
  assert rep.before_scatters == 6 and rep.after_scatters == 1

  got = single_pass_pileup(pos, bases, phred, rev, contig, _backend=backend)
  exp = (
    np.bincount(pos, weights=(bases == 0).astype(np.int64), minlength=contig).astype(np.int64),
    np.bincount(pos, weights=(bases == 1).astype(np.int64), minlength=contig).astype(np.int64),
    np.bincount(pos, weights=(bases == 2).astype(np.int64), minlength=contig).astype(np.int64),
    np.bincount(pos, weights=(bases == 3).astype(np.int64), minlength=contig).astype(np.int64),
    np.bincount(pos, weights=phred.astype(np.int64), minlength=contig).astype(np.int64),
    np.bincount(pos, weights=rev.astype(np.int64), minlength=contig).astype(np.int64),
  )
  for a, b in zip(got, exp):
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_bio_gallery_tanimoto_protein_sw_and_ssm(backend):
  rng = np.random.default_rng(23)

  # 3. Morgan 2048-bit Tanimoto screening with exact tie preservation
  q_fp = rng.integers(0, 2**63, size=(8, 32), dtype=np.uint64)
  l_fp = rng.integers(0, 2**63, size=(150, 32), dtype=np.uint64)
  q_fp[0] = 0
  l_fp[5] = l_fp[135] = q_fp[1]  # exact 1.0 tie across tile boundary

  lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
  pq = lut[q_fp.view(np.uint8)].reshape(8, 32, 8).sum(axis=(1, 2))[:, None]
  pl = lut[l_fp.view(np.uint8)].reshape(150, 32, 8).sum(axis=(1, 2))[None, :]
  inter = lut[(q_fp[:, None, :] & l_fp[None, :, :]).view(np.uint8)].reshape(8, 150, 32, 8).sum(axis=(2, 3))
  denom = pq + pl - inter
  sim = np.where(denom > 0, inter.astype(np.float32) / np.maximum(denom, 1).astype(np.float32), np.float32(0.0))
  exp_hits = np.argmax(sim, axis=1)
  got_hits = tanimoto_nearest_hit(q_fp, l_fp, _backend=backend)
  np.testing.assert_array_equal(got_hits, exp_hits)
  assert int(got_hits[1]) == 5

  # 4. 3D protein RBF smoothing (Law 1 flash_monoid) + coordinate gradient vs finite diff
  coords = rng.normal(size=(16, 3)).astype(np.float32) * 0.8
  feats = rng.normal(size=(16, 8)).astype(np.float32) * 0.4
  targets = rng.normal(size=(16, 8)).astype(np.float32) * 0.4
  gamma = np.float32(-0.2)

  rep_r = dsl.show_rewrite(protein_rbf_message_passing, coords, feats, gamma, backend="multicore")
  assert "flash_monoid" in rep_r.monoidal_plans
  d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * gamma
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  ref_rbf = (e / np.sum(e, axis=-1, keepdims=True)) @ feats
  np.testing.assert_allclose(
    protein_rbf_message_passing(coords, feats, gamma, _backend=backend), ref_rbf, rtol=1e-5, atol=1e-5
  )

  with pk.options(backend=backend):
    g_coords = protein_coord_grad(coords, feats, targets, gamma)
  eps = 1e-3
  fd = np.zeros_like(coords)
  for idx in np.ndindex(coords.shape):
    cp = coords.copy()
    cm = coords.copy()
    cp[idx] += eps
    cm[idx] -= eps
    fd[idx] = (
      float(protein_rbf_loss(cp, feats, targets, gamma)) - float(protein_rbf_loss(cm, feats, targets, gamma))
    ) / (2.0 * eps)
  np.testing.assert_allclose(g_coords, fd, rtol=5e-3, atol=5e-3)

  # 5a. Banded Smith-Waterman vs reference DP
  sa = rng.integers(0, 4, size=(48,), dtype=np.int32)
  sb = sa.copy()
  sb[10:15] = (sb[10:15] + 1) % 4
  got_sw = float(
    banded_smith_waterman(sa, sb, np.float32(2.0), np.float32(-1.0), np.float32(-2.0), 8, _backend=backend)
  )
  prev = [0.0] * (len(sb) + 1)
  curr = [0.0] * (len(sb) + 1)
  exp_sw = 0.0
  for i in range(1, len(sa) + 1):
    j_lo, j_hi = max(1, i - 8), min(len(sb), i + 8)
    for j in range(j_lo, j_hi + 1):
      sub = 2.0 if int(sa[i - 1]) == int(sb[j - 1]) else -1.0
      v = max(0.0, prev[j - 1] + sub, prev[j] - 2.0, curr[j - 1] - 2.0)
      curr[j] = v
      if v > exp_sw:
        exp_sw = v
    for j in range(j_lo, j_hi + 1):
      prev[j] = curr[j]
      curr[j] = 0.0
  assert abs(got_sw - exp_sw) < 1e-4

  # 5b. DNA SSM scan (Law 4 TileScan)
  decay = np.full((4096,), 0.98, dtype=np.float32)
  track = rng.normal(size=(4096,)).astype(np.float32)
  rep_s = dsl.show_rewrite(dna_ssm_scan, decay, track, backend="multicore")
  assert any("Law 4" in law for law in rep_s.laws_triggered)
  got_ssm = dna_ssm_scan(decay, track, _backend=backend)
  ref_ssm = np.zeros_like(track)
  st = np.float32(0.0)
  for t in range(4096):
    st = np.float32(decay[t] * st + track[t])
    ref_ssm[t] = st
  np.testing.assert_allclose(got_ssm, ref_ssm, rtol=1e-4, atol=1e-4)
