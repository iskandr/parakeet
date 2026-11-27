# Bioinformatics & structural biology

Scientific computing in genomics, cheminformatics, and structural biology is
full of kernels that do not fit standard dense GEMM libraries: byte-level FASTQ
parsing and 2-bit k-mer packing, multi-channel alignment pileups, 2048-bit
chemical fingerprint screening, 3D protein coordinate message passing, and
banded dynamic programming alignments.

Parakeet compiles these workloads directly from concise NumPy and data-parallel
operators into fused, multi-core CPU and GPU code without hand-writing C, CUDA,
or Metal.

Every snippet below is included from [`examples/bio_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/bio_gallery.py).
Run the gallery checks or reproducible benchmarks from a source checkout:

```bash
python -m examples.bio_gallery
python benchmarks/bio_workloads.py
```

## How algebraic primitives map to bio workloads

| Domain task | Operator / optimization in Parakeet | What the compiler eliminates |
|---|---|---|
| FASTQ k-mers & read QC | [`par.map`](operators.md#map) (multi-output) + `IntEnum` + ASCII-dict LUT | Explicit `parfor`, manual `np.empty` buffers, and ASCII bit-twiddling |
| Multi-channel BAM pileup | Keyed [`IndexScatter`](primitives.md#scatter) fusion | Redundant passes over alignment positions (`6 -> 1`) |
| Chemical fingerprint screening | `par.popcount` + [`par.allpairs`](operators.md#allpairs) + ordered [`par.fold`](operators.md#fold) (`pairwise_fold`) | O(N · M) similarity matrix allocation |
| 3D protein RBF / equivariant attention | [Online rescaled-monoid tiling](algebra.md#law-1-rescaled-semiring-monoids) + `par.grad` | O(N²) pairwise distance & weight matrices |
| Banded Smith-Waterman & genome SSMs | Two-row banded DP + [chunkwise affine prefix scan](algebra.md#law-4-chunkwise-state-space-duality) (`TileScan`) | Python DP interpreter overhead & sequential scan bottlenecks |

## 1. FASTQ 2-bit canonical k-mers and read QC

Parakeet accepts `list[str]` directly (packing equal-length strings into a 2D
`uint8` array or variable-length strings into a `RaggedArray[uint8]`), compiles
constant single-character dictionaries (`BASE_CODE`) into 256-byte lookup
tables, supports `IntEnum` alphabets (`Base.A`, `Base.C`, `Base.G`, `Base.T`),
and unboxes multi-output `par.map` tuples into one parallel pass:

```python
--8<-- "examples/bio_gallery.py:fastq_kmer_gc"
```

## 2. Single-pass multi-channel alignment pileup

A pileup records A/C/G/T base counts, Phred quality sums, and reverse-strand
counts at each reference coordinate. Because all six `par.scatter_add`
updates share the same index array `ref_pos`, Parakeet's scatter-fusion pass
combines them into a single privatized traversal (`IndexScatter: 6 -> 1`):

```python
--8<-- "examples/bio_gallery.py:bam_pileup_fused"
```

## 3. 2048-bit Morgan fingerprint Tanimoto screening

Chemical similarity screening compares query fingerprints `(N, 32)` against a
compound library `(M, 32)` of packed `uint64` bitvectors (2,048 bits each).
Combining `par.popcount` with `par.allpairs` and an ordered `par.fold` finds
the closest library compound per query while preserving first-index ties:

```python
--8<-- "examples/bio_gallery.py:tanimoto_screen"
```

## 4. 3D protein coordinate RBF message passing

Pairwise radial-basis-function smoothing over 3D residue coordinates `(N, 3)`
uses the same [rescaled-monoid](algebra.md#law-1-rescaled-semiring-monoids)
structure as softmax attention. Online monoidal tiling fuses the
squared-distance computation, softmax normalization, and feature aggregation
into a single tiled pass with zero O(N²) score allocation, and `par.grad`
differentiates through the 3D coordinates directly:

```python
--8<-- "examples/bio_gallery.py:protein_rbf"
```

## 5. Banded Smith-Waterman alignment and DNA state-space scan

Compiled two-row banded Smith-Waterman keeps only O(M) score scratch in L1
cache, while 1D DNA exponential state-space recurrences automatically
rewrite into [chunkwise parallel affine prefix scans](algebra.md#law-4-chunkwise-state-space-duality) (`TileScan`):

```python
--8<-- "examples/bio_gallery.py:banded_sw_and_ssm"
```

## 6. Classic sequence & structural algorithms (`examples/bio_algorithms.py`)

Beyond fused gallery kernels, [`examples/bio_algorithms.py`](https://github.com/iskandr/parakeet/blob/master/examples/bio_algorithms.py)
implements seven textbook bioinformatics kernels in compiled Python/NumPy:

* **3-state affine-gap Gotoh global alignment + traceback (`gotoh_global_align`):**
  Maintains `M`, `Ix`, and `Iy` DP matrices plus packed traceback directions and
  emits run-length-encoded CIGAR `(ops, lengths)` via local growable lists.
* **Multi-word Myers bit-vector edit distance (`myers_edit_distance`):**
  Bit-parallel Levenshtein distance over arbitrary pattern lengths using 64-bit
  `uint64` word blocks (`Pv`, `Mv`, `Peq`).
* **O(N) monotone-deque minimizer sketching (`minimizer_sketch`):**
  Canonical 2-bit k-mers hashed with SplitMix64 and sketched in a single linear
  pass using a circular monotone deque.
* **FM-index / BWT backward search (`build_fm_index`, `fm_backward_search_batch`):**
  `C` array + sampled `Occ` checkpoints + LF-mapping queried in parallel over a
  `RaggedArray[uint8]` batch of reads.
* **CIGAR expansion (`expand_cigar`):**
  One-line parallel expansion of `(ops, lengths)` via `par.flatmap`
  (`IndexExpand`).
* **Profile HMM Viterbi + Forward-Backward (`hmm_viterbi_and_posterior`):**
  Log-space Viterbi decoding plus Forward-Backward posterior state probabilities.
* **Kabsch 3D superposition & RMSD (`kabsch_rmsd`, `batched_kabsch_rmsd`):**
  Uses Parakeet's compiled `np.linalg.svd` and `np.linalg.det` (with proper
  rotation reflection correction when `det(R) < 0`) inside `par.parfor` to
  superimpose thousands of protein structures in parallel with zero external
  LAPACK dependency.

## 7. Long-read RNA alignment & quantification (`examples/patro_longread_rna.py`)

Inspired by [Rob Patro](https://www.cs.umd.edu/~rob/)'s
[COMBINE lab](https://combine-lab.github.io/) tools (SSHash, Piscem, Spearfish,
and Oarfish), [`examples/patro_longread_rna.py`](https://github.com/iskandr/parakeet/blob/master/examples/patro_longread_rna.py)
puts a complete long-read RNA aligner and isoform quantifier into ~250 lines of
compiled Python/NumPy:

1. **Unitig collapse + SSHash sparse minimizer index (`build_unitig_sshash_index`):**
   Collapses redundant gene isoforms into exact maximal shared unitigs stored
   in CSR arrays with `unitig -> (tx_id, tx_pos)` postings and a
   `FrozenHashTable` mapping sparse `m`-mer minimizers to small unitig buckets.
2. **Streaming unitig contiguity + Spearfish delta-mapping (`stream_map_long_reads`):**
   Inside `par.parfor` over a `RaggedArray[uint8]` of long reads, once a
   minimizer probe lands on a unitig, the inner loop extends the Maximal Exact
   Match (MEM) along that unitig in O(1) per base with zero hash lookups,
   projects the anchor to all sharing isoforms via CSR postings, chains
   co-linear diagonals, and retains competitive isoforms within `delta_frac` of
   the top score.
3. **Oarfish-style coverage-aware EM (`oarfish_em_quantify`):**
   Runs compiled Expectation-Maximization over the delta-mapped compatibility
   sets, weighting each compatible isoform by alignment score and effective
   length compatibility.

On 80 real human Ensembl cDNA isoforms (`TP53`, `PKM`, `MAPT`, `ACTB`;
225,539 raw bp), unitig collapsing shrinks the reference by **9.7x**
(`23,247` unitig bp across 231 unitigs), the SSHash fast path skips the hash
table on **92.2%** of base steps, multicore mapping sustains **548.6 Mbp/s** on
Apple M1 Pro (11-replicate median), and Oarfish EM achieves **Pearson r = 0.965**
against ground truth.

## 8. Overlapped CPU I/O, decompression, and compiler scope

Parakeet intentionally does not embed POSIX file descriptors, zlib/zstd
container parsers, or streaming I/O syscalls inside `@par.jit`. Keeping I/O out
of whole-program compilation keeps the compiler small, portable, and GPU-safe,
while still allowing full overlap between disk/codec work and compute:

### Double-buffered CPU I/O + kernel orchestration

Compiled Parakeet functions **release the Python GIL** for their entire C,
multicore, CUDA, or Metal execution. That makes overlapped streaming
straightforward with a 2-slot (or N-slot) ring buffer:

1. **Producer thread (`ThreadPoolExecutor` / background reader):** Reads raw
   FASTQ/BAM chunks from disk or `mmap`, decompresses them with a fast native
   codec (`isal`, `zlib-ng`, `libdeflate`, or `pysam`), and packs the next slab
   into a pre-allocated pinned `uint8` / `RaggedArray` host buffer (and on GPU,
   stages an asynchronous host-to-device copy onto slot `k % 2`).
2. **Consumer thread (Parakeet compute):** Launches `@par.jit` kernels (e.g.,
   `par.split_bytes`, `encode_canonical_kmers_and_gc`, `stream_map_long_reads`,
   or `single_pass_pileup`) on slab `(k - 1) % 2`. Because the GIL is released,
   CPU decompression of slab `k` overlaps directly with multicore/GPU kernels
   crunching slab `k - 1`.
3. **Thread budget coordination:** On CPU-only hosts, reserve 1-2 cores for the
   decompression producer and cap Parakeet's worker pool (`PARAKEET_THREADS`)
   to the remaining cores so reader and compute threads do not oversubscribe.

### What belongs in Parakeet vs. outside its semantic hull

* **Fits inside `@par.jit`:** *Block-parallel BGZF DEFLATE inflate.* Because
  BGZF splits `.bam` / `.fastq.gz` files into independent `<= 64 KiB` compressed
  blocks whose uncompressed sizes (`ISIZE`) are known up front from the block
  headers, [`examples/bgzf_and_streaming_io.py`](https://github.com/iskandr/parakeet/blob/master/examples/bgzf_and_streaming_io.py)
  compiles a full RFC-1951 DEFLATE inflater (stored, fixed Huffman, and dynamic
  Huffman with a 9-bit fast LUT + LZ77) in pure Python across all BGZF blocks
  in parallel, alongside a double-buffered overlapped `zlib` + Parakeet FASTQ
  pipeline. On 20,000 FASTQ reads (6.31 MB raw / 3.16 MB BGZF, 97 blocks) on
  Apple M1 Pro:
  - Raw inflate: Python `gzip` (C `zlib` 1T) **226 MB/s** vs. Parakeet BGZF
    1T **203 MB/s** vs. Parakeet BGZF multicore **1,579 MB/s** (7.0x faster).
  - End-to-end decompress + FASTQ QC: sequential `gzip` + QC **168 MB/s** vs.
    overlapped double-buffered `zlib` + QC **341 MB/s** (2.0x) vs. Parakeet
    parallel BGZF + QC **906 MB/s** (5.4x).
* **Escapes Parakeet's semantic hull:** Single-stream sequential `gzip`/`zstd`
  decompression, container framing/CRC error recovery, file I/O syscalls, and
  unbounded dynamic output growth on GPU. Single-stream DEFLATE has a serial
  32 KiB sliding-window dependency with no data parallelism, and native
  SIMD/assembly libraries (`isa-l`, `libdeflate`, `zstd`) already saturate NVMe
  bandwidth on 1-2 CPU cores. Feed decompressed `uint8` slabs from those
  libraries into Parakeet instead.

## Inspecting the rewrites

Run `par.show_rewrite` on any workload above to verify that scatter fusion,
online monoidal tiling, or chunkwise parallel scans fired:

```python
import numpy as np
import parakeet as par
from examples.bio_gallery import single_pass_pileup, protein_rbf_message_passing

pos = np.zeros(10_000, np.int64)
u8 = np.zeros(10_000, np.uint8)
print(par.show_rewrite(single_pass_pileup, pos, u8, u8, u8, 4096).summary())

coords = np.zeros((128, 3), np.float32)
feats = np.zeros((128, 16), np.float32)
print(par.show_rewrite(protein_rbf_message_passing, coords, feats, np.float32(-0.05)).summary())
```
