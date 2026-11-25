# Bioinformatics & structural biology

Scientific computing in genomics, cheminformatics, and structural biology is
full of kernels that do not fit standard dense GEMM libraries: byte-level FASTQ
parsing and 2-bit k-mer packing, multi-channel alignment pileups, 2048-bit
chemical fingerprint screening, 3D protein coordinate message passing, and
banded dynamic programming alignments.

Parakeet compiles these workloads directly from concise NumPy and data-parallel
operators into fused, multi-core CPU and GPU code — without hand-writing C,
CUDA, or Metal.

Every snippet below is included from [`examples/bio_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/bio_gallery.py).
Run the gallery checks or reproducible benchmarks from a source checkout:

```bash
python -m examples.bio_gallery
python benchmarks/bio_workloads.py
```

## How algebraic primitives map to bio workloads

| Domain task | Algebraic operator / law in Parakeet | What the compiler eliminates |
|---|---|---|
| FASTQ k-mers & read QC | `parfor` + `uint8` / `uint64` bitwise arithmetic | Python per-character loops and string copies |
| Multi-channel BAM pileup | Keyed `IndexScatter` fusion | Redundant passes over alignment positions (`6 -> 1`) |
| Chemical fingerprint screening | `pk.allpairs` + ordered `pk.fold` (`pairwise_fold`) | O(N · M) similarity matrix allocation |
| 3D protein RBF / equivariant attention | Law 1 rescaled-monoid tiling (`flash_monoid`) + `grad` | O(N²) pairwise distance & weight matrices |
| Banded Smith-Waterman & genome SSMs | Two-row banded DP + Law 4 `TileScan` | Python DP interpreter overhead & sequential scan bottlenecks |

## 1. FASTQ 2-bit canonical k-mers and read QC

Representing raw DNA and Phred quality bytes as `uint8` arrays lets Parakeet
compute forward and reverse-complement 2-bit `uint64` canonical k-mers
(`min(fwd, revcomp)`), GC fraction, and mean Phred quality in one parallel pass
over reads:

```python
--8<-- "examples/bio_gallery.py:fastq_kmer_gc"
```

## 2. Single-pass multi-channel alignment pileup

A pileup records A/C/G/T base counts, Phred quality sums, and reverse-strand
counts at each reference coordinate. Because all six `parakeet.scatter_add`
updates share the same index array `ref_pos`, Parakeet's scatter-fusion pass
combines them into a single privatized traversal (`IndexScatter: 6 -> 1`):

```python
--8<-- "examples/bio_gallery.py:bam_pileup_fused"
```

## 3. 2048-bit Morgan fingerprint Tanimoto screening

Chemical similarity screening compares query fingerprints `(N, 32)` against a
compound library `(M, 32)` of packed `uint64` bitvectors (2,048 bits each).
Combining a portable 64-bit SWAR popcount with `pk.allpairs` and an ordered
`pk.fold` finds the closest library compound per query while preserving
first-index ties:

```python
--8<-- "examples/bio_gallery.py:tanimoto_screen"
```

## 4. 3D protein coordinate RBF message passing

Pairwise radial-basis-function smoothing over 3D residue coordinates `(N, 3)`
uses the same rescaled-monoid structure as softmax attention. Law 1 fuses the
squared-distance computation, softmax normalization, and feature aggregation
into a single tiled pass with zero O(N²) score allocation — and `parakeet.grad`
differentiates through the 3D coordinates directly:

```python
--8<-- "examples/bio_gallery.py:protein_rbf"
```

## 5. Banded Smith-Waterman alignment and DNA state-space scan

Compiled two-row banded Smith-Waterman keeps only O(M) score scratch in L1
cache, while 1D DNA exponential state-space recurrences automatically
rewrite into Law 4 chunkwise parallel affine prefix scans (`TileScan`):

```python
--8<-- "examples/bio_gallery.py:banded_sw_and_ssm"
```

## Inspecting the rewrites

Run `pk.show_rewrite` on any workload above to verify that scatter fusion,
Law 1 monoidal tiling, or Law 4 chunkwise scans fired:

```python
import numpy as np
import parakeet as pk
from examples.bio_gallery import single_pass_pileup, protein_rbf_message_passing

pos = np.zeros(10_000, np.int64)
u8 = np.zeros(10_000, np.uint8)
print(pk.show_rewrite(single_pass_pileup, pos, u8, u8, u8, 4096).summary())

coords = np.zeros((128, 3), np.float32)
feats = np.zeros((128, 16), np.float32)
print(pk.show_rewrite(protein_rbf_message_passing, coords, feats, np.float32(-0.05)).summary())
```
