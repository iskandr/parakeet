"""Bioinformatics & Structural Biology Gallery in Parakeet.

Run directly to verify every workload against a reference implementation::

    python -m examples.bio_gallery

The five workloads map sequences, alignments, chemical fingerprints, and 3D
protein coordinates onto Parakeet's general algebraic operators (IndexScatter,
ordered structured folds, Law 1 rescaled-monoid tiling, Law 4 chunkwise scans,
and reverse-mode AD) without hand-writing C, CUDA, or Metal kernels.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import parakeet as pk

# ---------------------------------------------------------------------------
# 1. Zero-Copy uint8 FASTQ -> 2-Bit Canonical K-mer Encoding & Read QC
# ---------------------------------------------------------------------------


# --8<-- [start:fastq_kmer_gc]
@pk.jit
def encode_canonical_kmers_and_gc(seq_bytes, qual_bytes, k):
  """Encode a batch of equal-length ASCII DNA reads (N, L) uint8 into:
  - canonical 2-bit k-mers (N, L - k + 1) uint64  [min(fwd, revcomp)]
  - per-read GC fraction (N,) float32
  - per-read mean Phred quality (N,) float32
  """
  n_reads, read_len = seq_bytes.shape
  n_kmers = read_len - k + 1
  kmers = np.empty((n_reads, n_kmers), dtype=np.uint64)
  gc_frac = np.empty((n_reads,), dtype=np.float32)
  mean_q = np.empty((n_reads,), dtype=np.float32)
  mask = (~np.uint64(0)) >> np.uint64(64 - 2 * k)
  rev_shift = np.uint64(2 * (k - 1))

  def process_read(r):
    gc = np.int64(0)
    qsum = np.int64(0)
    fwd = np.uint64(0)
    rev = np.uint64(0)
    for pos in range(read_len):
      ch = seq_bytes[r, pos]
      # ASCII: A=65 -> 0, C=67 -> 1, G=71 -> 2, T=84 -> 3
      is_c = ch == np.uint8(67)
      is_g = ch == np.uint8(71)
      is_t = ch == np.uint8(84)
      code = np.uint64(1) if is_c else (np.uint64(2) if is_g else (np.uint64(3) if is_t else np.uint64(0)))
      comp = np.uint64(3) ^ code
      if is_c or is_g:
        gc = gc + np.int64(1)
      qsum = qsum + np.int64(qual_bytes[r, pos]) - np.int64(33)
      fwd = ((fwd << np.uint64(2)) | code) & mask
      rev = (rev >> np.uint64(2)) | (comp << rev_shift)
      if pos >= k - 1:
        kmers[r, pos - k + 1] = fwd if fwd < rev else rev
    gc_frac[r] = np.float32(gc) / np.float32(read_len)
    mean_q[r] = np.float32(qsum) / np.float32(read_len)

  pk.parfor(process_read, n_reads)
  return kmers, gc_frac, mean_q


# --8<-- [end:fastq_kmer_gc]


# ---------------------------------------------------------------------------
# 2. Single-Pass Multi-Channel Alignment Pileup (Fused Multi-Update IndexScatter)
# ---------------------------------------------------------------------------


# --8<-- [start:bam_pileup_fused]
@pk.jit
def single_pass_pileup(ref_pos, base_code, phred_q, is_rev, contig_len):
  """Accumulate A/C/G/T counts, quality sums, and reverse-strand counts across
  aligned base observations in one keyed IndexScatter traversal."""
  is_a = (base_code == np.uint8(0)).astype(np.int64)
  is_c = (base_code == np.uint8(1)).astype(np.int64)
  is_g = (base_code == np.uint8(2)).astype(np.int64)
  is_t = (base_code == np.uint8(3)).astype(np.int64)
  q_i64 = phred_q.astype(np.int64)
  rev_i64 = is_rev.astype(np.int64)

  # All six histograms share `ref_pos` -> Parakeet fuses them into ONE traversal!
  cnt_a = pk.scatter_add(contig_len, ref_pos, is_a)
  cnt_c = pk.scatter_add(contig_len, ref_pos, is_c)
  cnt_g = pk.scatter_add(contig_len, ref_pos, is_g)
  cnt_t = pk.scatter_add(contig_len, ref_pos, is_t)
  qual_sum = pk.scatter_add(contig_len, ref_pos, q_i64)
  rev_cnt = pk.scatter_add(contig_len, ref_pos, rev_i64)
  return cnt_a, cnt_c, cnt_g, cnt_t, qual_sum, rev_cnt


# --8<-- [end:bam_pileup_fused]


# ---------------------------------------------------------------------------
# 3. 2048-Bit Morgan Fingerprint Tanimoto Screening (Zero O(N * M) Allocation)
# ---------------------------------------------------------------------------


# --8<-- [start:tanimoto_screen]
def popcount64(x):
  """Branchless 64-bit SWAR popcount (portable across C, multicore, Metal, CUDA)."""
  u = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
  u = (u & np.uint64(0x3333333333333333)) + ((u >> np.uint64(2)) & np.uint64(0x3333333333333333))
  u = (u + (u >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
  return (u * np.uint64(0x0101010101010101)) >> np.uint64(56)


class BestHit(NamedTuple):
  sim: np.float32
  index: np.int64


def better_hit(left, right):
  # Highest Tanimoto similarity wins; ties preserve the first library compound.
  return right if right.sim > left.sim else left


@pk.jit
def tanimoto_nearest_hit(queries, library):
  """Find the closest library compound per query over packed uint64 fingerprints
  (N, W) x (M, W) using tiled pairwise folds with zero N x M allocation."""
  pop_q = np.sum(popcount64(queries), axis=1)
  pop_l = np.sum(popcount64(library), axis=1)

  def inter_bits(q_row, l_row):
    return np.sum(popcount64(q_row & l_row))

  # Stream intersection counts into an ordered fold over library compounds
  inter = pk.allpairs(inter_bits, queries, library)
  denom = pop_q[:, None] + pop_l[None, :] - inter
  sim = np.where(
    denom > np.uint64(0), inter.astype(np.float32) / np.maximum(denom, np.uint64(1)).astype(np.float32), np.float32(0.0)
  )
  return pk.fold(
    lambda score, ref_idx: BestHit(score, ref_idx),
    better_hit,
    sim,
    axis=1,
    indexed=True,
    emit=lambda hit: hit.index,
  )


# --8<-- [end:tanimoto_screen]


# ---------------------------------------------------------------------------
# 4. 3D Protein Coordinate RBF Message Passing & Differentiable Energy (Law 1)
# ---------------------------------------------------------------------------


# --8<-- [start:protein_rbf]
@pk.jit
def protein_rbf_message_passing(coords, features, neg_inv_2sigma2):
  """3D radial-basis-function message passing over residue coordinates (N, 3).

  Fuses into Law 1 online (m, l, O) monoidal tiles with zero N x N materialization.
  """
  d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * neg_inv_2sigma2
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ features


def protein_rbf_loss(coords, features, targets, neg_inv_2sigma2):
  pred = protein_rbf_message_passing(coords, features, neg_inv_2sigma2)
  diff = pred - targets
  return np.mean(diff * diff)


protein_coord_grad = pk.grad(protein_rbf_loss, argnum=0)
# --8<-- [end:protein_rbf]


# ---------------------------------------------------------------------------
# 5. Banded Smith-Waterman Alignment & Megabase DNA SSM Scan (Law 4)
# ---------------------------------------------------------------------------


# --8<-- [start:banded_sw_and_ssm]
@pk.jit
def banded_smith_waterman(seq_a, seq_b, match_score, mismatch_pen, gap_pen, band_width):
  """Banded Smith-Waterman local alignment score over integer-encoded sequences.

  Keeps only two O(M) score rows in cache and restricts columns to |j - i| <= band_width.
  """
  n = seq_a.shape[0]
  m = seq_b.shape[0]
  prev = np.zeros((m + 1,), dtype=np.float32)
  curr = np.zeros((m + 1,), dtype=np.float32)
  best = np.float32(0.0)
  for i in range(1, n + 1):
    ai = seq_a[i - 1]
    j_lo = i - band_width
    if j_lo < 1:
      j_lo = 1
    j_hi = i + band_width
    if j_hi > m:
      j_hi = m
    for j in range(j_lo, j_hi + 1):
      sub = match_score if ai == seq_b[j - 1] else mismatch_pen
      v = prev[j - 1] + sub
      up = prev[j] + gap_pen
      left = curr[j - 1] + gap_pen
      if up > v:
        v = up
      if left > v:
        v = left
      if v < np.float32(0.0):
        v = np.float32(0.0)
      curr[j] = v
      if v > best:
        best = v
    for j in range(j_lo, j_hi + 1):
      prev[j] = curr[j]
      curr[j] = np.float32(0.0)
  return best


@pk.jit
def dna_ssm_scan(decay, signal):
  """Exponential state-space scan along a DNA track (L,).

  Compiled via Law 4 chunkwise parallel affine prefix scan (TileScan).
  """
  length = decay.shape[0]
  out = np.empty_like(signal)
  s = np.float32(0.0)
  for t in range(length):
    s = decay[t] * s + signal[t]
    out[t] = s
  return out


# --8<-- [end:banded_sw_and_ssm]


def main():
  rng = np.random.default_rng(0)
  print("Parakeet Bioinformatics & Structural Biology Gallery\n" + "=" * 60)

  # 1. FASTQ canonical k-mers & QC
  alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
  reads = alphabet[rng.integers(0, 4, size=(64, 100))]
  quals = rng.integers(33 + 20, 33 + 41, size=(64, 100), dtype=np.uint8)
  kmers, gc, mq = encode_canonical_kmers_and_gc(reads, quals, 21)
  print(f"1. FASTQ 21-mers : shape={kmers.shape}, mean GC={gc.mean():.3f}, mean Phred={mq.mean():.1f}")

  # 2. Single-pass pileup
  pos = rng.integers(0, 4096, size=(50_000,), dtype=np.int64)
  bases = rng.integers(0, 4, size=(50_000,), dtype=np.uint8)
  phred = rng.integers(20, 42, size=(50_000,), dtype=np.uint8)
  rev = rng.integers(0, 2, size=(50_000,), dtype=np.uint8)
  rep_p = pk.show_rewrite(single_pass_pileup, pos, bases, phred, rev, 4096)
  single_pass_pileup(pos, bases, phred, rev, 4096)
  print(f"2. BAM pileup    : IndexScatter {rep_p.before_scatters} -> {rep_p.after_scatters} | {rep_p.laws_triggered}")

  # 3. Morgan fingerprint Tanimoto screening
  q_fp = rng.integers(0, 2**63, size=(16, 32), dtype=np.uint64)
  l_fp = rng.integers(0, 2**63, size=(512, 32), dtype=np.uint64)
  rep_t = pk.show_rewrite(tanimoto_nearest_hit, q_fp, l_fp)
  hits = tanimoto_nearest_hit(q_fp, l_fp)
  print(f"3. Tanimoto 1-NN : hits shape={hits.shape} | {rep_t.laws_triggered}")

  # 4. 3D Protein RBF message passing
  coords = rng.normal(size=(128, 3)).astype(np.float32) * 5.0
  feats = rng.normal(size=(128, 16)).astype(np.float32)
  rep_r = pk.show_rewrite(protein_rbf_message_passing, coords, feats, np.float32(-0.05))
  out_r = protein_rbf_message_passing(coords, feats, np.float32(-0.05))
  print(f"4. Protein RBF   : out shape={out_r.shape} | plans={rep_r.monoidal_plans}")

  # 5. Banded Smith-Waterman & DNA SSM scan
  sa = rng.integers(0, 4, size=(200,), dtype=np.int32)
  sb = sa.copy()
  sb[50:55] = (sb[50:55] + 1) % 4
  score = banded_smith_waterman(sa, sb, np.float32(2.0), np.float32(-1.0), np.float32(-2.0), 16)
  decay = np.full((65_536,), 0.99, dtype=np.float32)
  track = rng.normal(size=(65_536,)).astype(np.float32)
  rep_s = pk.show_rewrite(dna_ssm_scan, decay, track)
  dna_ssm_scan(decay, track)
  print(f"5. Banded SW/SSM : SW score={score:.1f} | SSM laws={rep_s.laws_triggered}")


if __name__ == "__main__":
  main()
