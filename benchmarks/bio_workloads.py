"""Reproducible benchmarks for bioinformatics & structural biology workloads.

Follows the benchmark discipline of ``benchmarks/keops_pairwise.py``:
- matched thread count (``OMP_NUM_THREADS`` / Parakeet ``multicore``)
- output verification against a reference implementation before timing
- warm interleaved repeats reporting median milliseconds and speedup
- explicit reporting when a NumPy baseline wins at small shapes

Run::

    python benchmarks/bio_workloads.py
"""

from __future__ import annotations

import os
import time

import numpy as np

from examples.bio_gallery import (
  banded_smith_waterman,
  dna_ssm_scan,
  encode_canonical_kmers_and_gc,
  protein_rbf_message_passing,
  single_pass_pileup,
  tanimoto_nearest_hit,
)


def _median_ms(fn_a, fn_b, repeats: int = 15, warmup: int = 3) -> tuple[float, float]:
  for _ in range(warmup):
    fn_a()
    fn_b()
  ta: list[float] = []
  tb: list[float] = []
  for r in range(repeats):
    if r % 2 == 0:
      t0 = time.perf_counter()
      fn_a()
      ta.append((time.perf_counter() - t0) * 1e3)
      t0 = time.perf_counter()
      fn_b()
      tb.append((time.perf_counter() - t0) * 1e3)
    else:
      t0 = time.perf_counter()
      fn_b()
      tb.append((time.perf_counter() - t0) * 1e3)
      t0 = time.perf_counter()
      fn_a()
      ta.append((time.perf_counter() - t0) * 1e3)
  return float(np.median(ta)), float(np.median(tb))


def _np_pileup(pos, bases, phred, rev, n):
  return (
    np.bincount(pos, weights=(bases == 0).astype(np.int64), minlength=n).astype(np.int64),
    np.bincount(pos, weights=(bases == 1).astype(np.int64), minlength=n).astype(np.int64),
    np.bincount(pos, weights=(bases == 2).astype(np.int64), minlength=n).astype(np.int64),
    np.bincount(pos, weights=(bases == 3).astype(np.int64), minlength=n).astype(np.int64),
    np.bincount(pos, weights=phred.astype(np.int64), minlength=n).astype(np.int64),
    np.bincount(pos, weights=rev.astype(np.int64), minlength=n).astype(np.int64),
  )


def _np_rbf(coords, feats, neg_gamma):
  d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * neg_gamma
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ feats


def _np_tanimoto_1nn(q, lib):
  lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

  def pop64(a):
    return lut[a.view(np.uint8)].reshape(*a.shape, 8).sum(axis=-1)

  pq = pop64(q).sum(axis=-1)[:, None]
  pl = pop64(lib).sum(axis=-1)[None, :]
  inter = pop64(q[:, None, :] & lib[None, :, :]).sum(axis=-1)
  denom = pq + pl - inter
  sim = np.where(denom > 0, inter.astype(np.float32) / np.maximum(denom, 1).astype(np.float32), np.float32(0.0))
  return np.argmax(sim, axis=1)


def _py_banded_sw(seq_a, seq_b, match_s, mismatch_p, gap_p, bw):
  n, m = len(seq_a), len(seq_b)
  prev = [0.0] * (m + 1)
  curr = [0.0] * (m + 1)
  best = 0.0
  for i in range(1, n + 1):
    ai = int(seq_a[i - 1])
    j_lo = max(1, i - bw)
    j_hi = min(m, i + bw)
    for j in range(j_lo, j_hi + 1):
      sub = match_s if ai == int(seq_b[j - 1]) else mismatch_p
      v = max(0.0, prev[j - 1] + sub, prev[j] + gap_p, curr[j - 1] + gap_p)
      curr[j] = v
      if v > best:
        best = v
    for j in range(j_lo, j_hi + 1):
      prev[j] = curr[j]
      curr[j] = 0.0
  return best


def main():
  rng = np.random.default_rng(42)
  threads = os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 1))
  print(f"Bioinformatics Benchmark Suite (threads={threads}, warm interleaved medians)")
  print(f"{'Workload':<44} {'Parakeet (ms)':>14} {'Baseline (ms)':>14} {'Speedup':>10}")
  print("-" * 86)

  # 1. Single-pass 6-channel pileup (500k bases -> 8192 contig bins)
  pos = rng.integers(0, 8192, size=(500_000,), dtype=np.int64)
  bases = rng.integers(0, 4, size=(500_000,), dtype=np.uint8)
  phred = rng.integers(20, 42, size=(500_000,), dtype=np.uint8)
  rev = rng.integers(0, 2, size=(500_000,), dtype=np.uint8)
  pk_p = single_pass_pileup(pos, bases, phred, rev, 8192)
  np_p = _np_pileup(pos, bases, phred, rev, 8192)
  for a, b in zip(pk_p, np_p):
    np.testing.assert_array_equal(a, b)
  t_pk, t_bl = _median_ms(
    lambda: single_pass_pileup(pos, bases, phred, rev, 8192),
    lambda: _np_pileup(pos, bases, phred, rev, 8192),
  )
  print(f"{'6-channel BAM pileup (500k -> 8192)':<44} {t_pk:14.2f} {t_bl:14.2f} {t_bl / t_pk:9.2f}x")

  # 2. 3D Protein RBF message passing (N=1024 residues, D=32)
  coords = rng.normal(size=(1024, 3)).astype(np.float32) * 8.0
  feats = rng.normal(size=(1024, 32)).astype(np.float32)
  gamma = np.float32(-0.02)
  np.testing.assert_allclose(
    protein_rbf_message_passing(coords, feats, gamma), _np_rbf(coords, feats, gamma), rtol=1e-4, atol=1e-4
  )
  t_pk, t_bl = _median_ms(
    lambda: protein_rbf_message_passing(coords, feats, gamma),
    lambda: _np_rbf(coords, feats, gamma),
    repeats=11,
  )
  print(f"{'3D protein RBF smoothing (N=1024, D=32)':<44} {t_pk:14.2f} {t_bl:14.2f} {t_bl / t_pk:9.2f}x")

  # 3. 2048-bit Morgan fingerprint Tanimoto 1-NN (64 queries x 4096 compounds)
  q_fp = rng.integers(0, 2**63, size=(64, 32), dtype=np.uint64)
  l_fp = rng.integers(0, 2**63, size=(4096, 32), dtype=np.uint64)
  np.testing.assert_array_equal(tanimoto_nearest_hit(q_fp, l_fp), _np_tanimoto_1nn(q_fp, l_fp))
  t_pk, t_bl = _median_ms(
    lambda: tanimoto_nearest_hit(q_fp, l_fp),
    lambda: _np_tanimoto_1nn(q_fp, l_fp),
    repeats=7,
    warmup=2,
  )
  print(f"{'Morgan 2048-bit Tanimoto 1-NN (64x4096)':<44} {t_pk:14.2f} {t_bl:14.2f} {t_bl / t_pk:9.2f}x")

  # 4. Banded Smith-Waterman (L=1024, band=32)
  sa = rng.integers(0, 4, size=(1024,), dtype=np.int32)
  sb = sa.copy()
  sb[::19] = (sb[::19] + 1) % 4
  s_pk = float(banded_smith_waterman(sa, sb, np.float32(2.0), np.float32(-1.0), np.float32(-2.0), 32))
  s_py = _py_banded_sw(sa, sb, 2.0, -1.0, -2.0, 32)
  assert abs(s_pk - s_py) < 1e-4
  t_pk, t_bl = _median_ms(
    lambda: banded_smith_waterman(sa, sb, np.float32(2.0), np.float32(-1.0), np.float32(-2.0), 32),
    lambda: _py_banded_sw(sa, sb, 2.0, -1.0, -2.0, 32),
    repeats=9,
    warmup=2,
  )
  print(f"{'Banded Smith-Waterman (L=1024, bw=32)':<44} {t_pk:14.2f} {t_bl:14.2f} {t_bl / t_pk:9.2f}x")

  # 5. FASTQ 21-mer encoding + GC/Phred QC (4096 reads x 150 bp)
  alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
  reads = alphabet[rng.integers(0, 4, size=(4096, 150))]
  quals = rng.integers(53, 74, size=(4096, 150), dtype=np.uint8)
  encode_canonical_kmers_and_gc(reads, quals, 21)
  fq_times = []
  for _ in range(11):
    t0 = time.perf_counter()
    encode_canonical_kmers_and_gc(reads, quals, 21)
    fq_times.append((time.perf_counter() - t0) * 1e3)
  ms_fq = float(np.median(fq_times))
  mb_s = (reads.nbytes + quals.nbytes) / (ms_fq * 1e-3) / 1e6
  print(f"{'FASTQ 21-mer + QC (4096x150 bp)':<44} {ms_fq:14.2f} {'(throughput)':>14} {mb_s:7.1f} MB/s")

  # 6. DNA SSM scan (262,144 bp, Law 4 TileScan)
  decay = np.full((262_144,), 0.995, dtype=np.float32)
  track = rng.normal(size=(262_144,)).astype(np.float32)
  dna_ssm_scan(decay, track)
  ssm_times = []
  for _ in range(11):
    t0 = time.perf_counter()
    dna_ssm_scan(decay, track)
    ssm_times.append((time.perf_counter() - t0) * 1e3)
  ms_ssm = float(np.median(ssm_times))
  print(
    f"{'DNA SSM scan (262k bp, Law 4 TileScan)':<44} {ms_ssm:14.2f} {'(TileScan)':>14} {262.144 / ms_ssm:7.1f} Mbp/s"
  )


if __name__ == "__main__":
  main()
