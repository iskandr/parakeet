"""Bioinformatics & Structural Biology Gallery in Parakeet.

Run directly to verify every workload against a reference implementation::

    python -m examples.bio_gallery

The five workloads map sequences, alignments, chemical fingerprints, and 3D
protein coordinates onto Parakeet's general algebraic operators (IndexScatter,
ordered structured folds, online rescaled-monoid tiling, chunkwise scans,
and reverse-mode AD) without hand-writing C, CUDA, or Metal kernels.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numpy as np

import parakeet as par

# ---------------------------------------------------------------------------
# 1. FASTQ -> 2-Bit Canonical K-mer Encoding & Read QC
# ---------------------------------------------------------------------------


# --8<-- [start:fastq_kmer_gc]
class Base(IntEnum):
  A = 0
  C = 1
  G = 2
  T = 3


BASE_CODE = {"A": Base.A, "C": Base.C, "G": Base.G, "T": Base.T}


@par.jit
def encode_canonical_kmers_and_gc(reads, quals, k):
  """Encode ASCII DNA reads (``list[str]`` or ``(N, L)`` uint8) into:
  - canonical 2-bit k-mers ``(N, L - k + 1)`` uint64  [min(fwd, revcomp)]
  - per-read GC fraction ``(N,)`` float32
  - per-read mean Phred quality ``(N,)`` float32
  """
  mask = (~np.uint64(0)) >> np.uint64(64 - 2 * k)
  rev_shift = np.uint64(2 * (k - 1))

  def encode_read(seq, qual):
    codes = BASE_CODE[seq]
    gc = np.mean((codes == Base.C) | (codes == Base.G)).astype(np.float32)
    mean_q = np.mean(qual.astype(np.float32) - 33.0)
    out = np.empty(len(seq) - k + 1, dtype=np.uint64)
    fwd = np.uint64(0)
    rev = np.uint64(0)
    for pos, c in enumerate(codes):
      code = np.uint64(c)
      fwd = ((fwd << np.uint64(2)) | code) & mask
      rev = (rev >> np.uint64(2)) | ((np.uint64(3) ^ code) << rev_shift)
      if pos >= k - 1:
        out[pos - k + 1] = min(fwd, rev)
    return out, gc, mean_q

  return par.map(encode_read, reads, quals)


# --8<-- [end:fastq_kmer_gc]


# ---------------------------------------------------------------------------
# 2. Single-Pass Multi-Channel Alignment Pileup (Fused Multi-Update IndexScatter)
# ---------------------------------------------------------------------------


# --8<-- [start:bam_pileup_fused]
@par.jit
def single_pass_pileup(ref_pos, base_code, phred_q, is_rev, contig_len):
  """Accumulate A/C/G/T counts, quality sums, and reverse-strand counts across
  aligned base observations in one keyed IndexScatter traversal."""
  # All six histograms share ref_pos, so Parakeet fuses them into one pass.
  cnt_a = par.scatter_add(contig_len, ref_pos, (base_code == Base.A).astype(np.int64))
  cnt_c = par.scatter_add(contig_len, ref_pos, (base_code == Base.C).astype(np.int64))
  cnt_g = par.scatter_add(contig_len, ref_pos, (base_code == Base.G).astype(np.int64))
  cnt_t = par.scatter_add(contig_len, ref_pos, (base_code == Base.T).astype(np.int64))
  qual_sum = par.scatter_add(contig_len, ref_pos, phred_q.astype(np.int64))
  rev_cnt = par.scatter_add(contig_len, ref_pos, is_rev.astype(np.int64))
  return cnt_a, cnt_c, cnt_g, cnt_t, qual_sum, rev_cnt


# --8<-- [end:bam_pileup_fused]


# ---------------------------------------------------------------------------
# 3. 2048-Bit Morgan Fingerprint Tanimoto Screening (Zero O(N * M) Allocation)
# ---------------------------------------------------------------------------


# --8<-- [start:tanimoto_screen]
class BestHit(NamedTuple):
  sim: np.float32
  index: np.int64


def better_hit(left, right):
  # Highest Tanimoto similarity wins; ties preserve the first library compound.
  return right if right.sim > left.sim else left


@par.jit
def tanimoto_nearest_hit(queries, library):
  """Find the closest library compound per query over packed uint64 fingerprints
  (N, W) x (M, W) using tiled pairwise folds with zero N x M allocation."""
  pop_q = np.sum(par.popcount(queries), axis=1)
  pop_l = np.sum(par.popcount(library), axis=1)

  inter = par.allpairs(lambda q_row, l_row: np.sum(par.popcount(q_row & l_row)), queries, library)
  denom = pop_q[:, None] + pop_l[None, :] - inter
  sim = np.where(denom > 0, inter.astype(np.float32) / np.maximum(denom, 1).astype(np.float32), np.float32(0.0))
  return par.fold(BestHit, better_hit, sim, axis=1, indexed=True, emit=lambda hit: hit.index)


# --8<-- [end:tanimoto_screen]


# ---------------------------------------------------------------------------
# 4. 3D Protein Coordinate RBF Message Passing & Differentiable Energy
# ---------------------------------------------------------------------------


# --8<-- [start:protein_rbf]
@par.jit
def protein_rbf_message_passing(coords, features, neg_inv_2sigma2):
  """3D radial-basis-function message passing over residue coordinates (N, 3).

  Fuses into online (m, l, O) monoidal tiles with zero N x N materialization.
  """
  d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * neg_inv_2sigma2
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ features


def protein_rbf_loss(coords, features, targets, neg_inv_2sigma2):
  pred = protein_rbf_message_passing(coords, features, neg_inv_2sigma2)
  diff = pred - targets
  return np.mean(diff * diff)


protein_coord_grad = par.grad(protein_rbf_loss, argnum=0)
# --8<-- [end:protein_rbf]


# ---------------------------------------------------------------------------
# 5. Banded Smith-Waterman Alignment & Megabase DNA SSM Scan
# ---------------------------------------------------------------------------


# --8<-- [start:banded_sw_and_ssm]
@par.jit
def banded_smith_waterman(seq_a, seq_b, match_score, mismatch_pen, gap_pen, band_width):
  """Banded Smith-Waterman local alignment score over sequences.

  Keeps only two O(M) score rows in cache and restricts columns to |j - i| <= band_width.
  """
  n, m = len(seq_a), len(seq_b)
  prev = np.zeros(m + 1, dtype=np.float32)
  curr = np.zeros(m + 1, dtype=np.float32)
  best = np.float32(0.0)
  for i in range(1, n + 1):
    ai = seq_a[i - 1]
    j_lo = max(1, i - band_width)
    j_hi = min(m, i + band_width)
    for j in range(j_lo, j_hi + 1):
      sub = match_score if ai == seq_b[j - 1] else mismatch_pen
      v = max(np.float32(0.0), prev[j - 1] + sub, prev[j] + gap_pen, curr[j - 1] + gap_pen)
      curr[j] = v
      best = max(best, v)
    prev[j_lo : j_hi + 1] = curr[j_lo : j_hi + 1]
    curr[j_lo : j_hi + 1] = 0.0
  return best


@par.jit
def dna_ssm_scan(decay, signal):
  """Exponential state-space scan along a DNA track (L,).

  Compiled via chunkwise parallel affine prefix scan (TileScan).
  """
  out = np.empty_like(signal)
  s = np.float32(0.0)
  for t in range(len(decay)):
    s = decay[t] * s + signal[t]
    out[t] = s
  return out


# --8<-- [end:banded_sw_and_ssm]


def main():
  rng = np.random.default_rng(0)
  print("Parakeet Bioinformatics & Structural Biology Gallery\n" + "=" * 60)

  # 1. FASTQ canonical k-mers & QC (accepts list[str] directly)
  alphabet = np.array(list("ACGT"))
  reads = ["".join(row) for row in alphabet[rng.integers(0, 4, size=(64, 100))]]
  quals = ["".join(chr(q) for q in row) for row in rng.integers(33 + 20, 33 + 41, size=(64, 100))]
  kmers, gc, mq = encode_canonical_kmers_and_gc(reads, quals, 21)
  print(f"1. FASTQ 21-mers : shape={kmers.shape}, mean GC={gc.mean():.3f}, mean Phred={mq.mean():.1f}")

  # 2. Single-pass pileup
  pos = rng.integers(0, 4096, size=(50_000,), dtype=np.int64)
  bases = rng.integers(0, 4, size=(50_000,), dtype=np.uint8)
  phred = rng.integers(20, 42, size=(50_000,), dtype=np.uint8)
  rev = rng.integers(0, 2, size=(50_000,), dtype=np.uint8)
  rep_p = par.show_rewrite(single_pass_pileup, pos, bases, phred, rev, 4096)
  single_pass_pileup(pos, bases, phred, rev, 4096)
  print(f"2. BAM pileup    : IndexScatter {rep_p.before_scatters} -> {rep_p.after_scatters} | {rep_p.laws_triggered}")

  # 3. Morgan fingerprint Tanimoto screening
  q_fp = rng.integers(0, 2**63, size=(16, 32), dtype=np.uint64)
  l_fp = rng.integers(0, 2**63, size=(512, 32), dtype=np.uint64)
  rep_t = par.show_rewrite(tanimoto_nearest_hit, q_fp, l_fp)
  hits = tanimoto_nearest_hit(q_fp, l_fp)
  print(f"3. Tanimoto 1-NN : hits shape={hits.shape} | {rep_t.laws_triggered}")

  # 4. 3D Protein RBF message passing
  coords = rng.normal(size=(128, 3)).astype(np.float32) * 5.0
  feats = rng.normal(size=(128, 16)).astype(np.float32)
  rep_r = par.show_rewrite(protein_rbf_message_passing, coords, feats, np.float32(-0.05))
  out_r = protein_rbf_message_passing(coords, feats, np.float32(-0.05))
  print(f"4. Protein RBF   : out shape={out_r.shape} | plans={rep_r.monoidal_plans}")

  # 5. Banded Smith-Waterman & DNA SSM scan
  sa = rng.integers(0, 4, size=(200,), dtype=np.int32)
  sb = sa.copy()
  sb[50:55] = (sb[50:55] + 1) % 4
  score = banded_smith_waterman(sa, sb, np.float32(2.0), np.float32(-1.0), np.float32(-2.0), 16)
  decay = np.full((65_536,), 0.99, dtype=np.float32)
  track = rng.normal(size=(65_536,)).astype(np.float32)
  rep_s = par.show_rewrite(dna_ssm_scan, decay, track)
  dna_ssm_scan(decay, track)
  print(f"5. Banded SW/SSM : SW score={score:.1f} | SSM laws={rep_s.laws_triggered}")


if __name__ == "__main__":
  main()
